from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments
MAX_LENGTH = 384
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRAD_ACCUM = 2
LEARNING_RATE = 2e-05
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2

class PairTaskDataset(Dataset):

    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer: Any, task: str):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.task = task

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(str(row['review_text_for_model']), str(row['aspect_prompt']), max_length=MAX_LENGTH, truncation=True, padding=False)
        item = dict(encoded)
        item['labels'] = float(row[self.task])
        return item

class WeightedBinaryTrainer(Trainer):

    def __init__(self, pos_weight: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.pos_weight = float(pos_weight)

    def compute_loss(self, model: torch.nn.Module, inputs: dict[str, torch.Tensor], return_outputs: bool=False, **kwargs: Any):
        labels = inputs.pop('labels').float().view(-1)
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        weight = torch.tensor([self.pos_weight], dtype=logits.dtype, device=logits.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)
        return (loss, outputs) if return_outputs else loss

class BestStateInMemoryCallback(TrainerCallback):
    """Keep the minimum-dev-loss state without checkpoint serialization."""

    def __init__(self, patience: int):
        self.patience = int(patience)
        self.best_loss = float('inf')
        self.best_epoch: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None
        self.bad_epochs = 0

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        metrics = metrics or {}
        if 'eval_loss' not in metrics or model is None:
            return control
        loss = float(metrics['eval_loss'])
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = float(state.epoch) if state.epoch is not None else None
            self.best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                control.should_training_stop = True
        return control

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    pos = values >= 0
    output[pos] = 1.0 / (1.0 + np.exp(-values[pos]))
    exp_values = np.exp(values[~pos])
    output[~pos] = exp_values / (1.0 + exp_values)
    return output

def train_transformer_task(seed: int, task: str, train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed_tag = f'seed_{seed:04d}'
    cache_dir = INTERMEDIATE_ROOT / 'transformer_probabilities' / seed_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{task}.npz'
    meta_path = cache_dir / f'{task}_training_metadata.json'
    if cache_path.exists() and meta_path.exists():
        stored = np.load(cache_path)
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        return (stored['dev_prob'], stored['test_prob'], metadata)
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(HF_MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(HF_MODEL_DIR), num_labels=1, local_files_only=True)
    filtered_train = train_rows if task == 'y_mention' else [r for r in train_rows if int(r['y_mention']) == 1]
    filtered_dev = dev_rows if task == 'y_mention' else [r for r in dev_rows if int(r['y_mention']) == 1]
    n_positive = sum((int(row[task]) for row in filtered_train))
    n_negative = len(filtered_train) - n_positive
    if n_positive <= 0 or n_negative <= 0:
        raise RuntimeError(f'Transformer task lacks two classes: seed={seed}, task={task}')
    pos_weight = n_negative / n_positive
    train_dataset = PairTaskDataset(filtered_train, tokenizer, task)
    dev_selection_dataset = PairTaskDataset(filtered_dev, tokenizer, task)
    dev_full_dataset = PairTaskDataset(dev_rows, tokenizer, task)
    test_full_dataset = PairTaskDataset(test_rows, tokenizer, task)
    work_dir = TRAINING_WORK_ROOT / seed_tag / task
    if work_dir.exists():
        shutil.rmtree(work_dir)
    best_state_callback = BestStateInMemoryCallback(EARLY_STOPPING_PATIENCE)
    training_args = TrainingArguments(output_dir=str(work_dir), eval_strategy='epoch', save_strategy='no', learning_rate=LEARNING_RATE, per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM, num_train_epochs=EPOCHS, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO, load_best_model_at_end=False, logging_strategy='epoch', report_to=[], seed=seed, data_seed=seed, disable_tqdm=True, fp16=False, bf16=False, dataloader_num_workers=0)
    trainer = WeightedBinaryTrainer(pos_weight=pos_weight, model=model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_selection_dataset, processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True), callbacks=[best_state_callback])
    start = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - start
    if best_state_callback.best_state is None:
        raise RuntimeError(f'No dev checkpoint state captured: seed={seed}, task={task}')
    trainer.model.load_state_dict(best_state_callback.best_state, strict=True)
    dev_logits = np.asarray(trainer.predict(dev_full_dataset).predictions).reshape(-1)
    test_logits = np.asarray(trainer.predict(test_full_dataset).predictions).reshape(-1)
    dev_prob = sigmoid(dev_logits)
    test_prob = sigmoid(test_logits)
    eval_entries = [entry for entry in trainer.state.log_history if 'eval_loss' in entry]
    metadata = {'seed': seed, 'task': task, 'train_rows': len(filtered_train), 'dev_checkpoint_selection_rows': len(filtered_dev), 'dev_prediction_rows': len(dev_rows), 'test_prediction_rows': len(test_rows), 'positive_train_rows': n_positive, 'negative_train_rows': n_negative, 'pos_weight': pos_weight, 'train_runtime_seconds': elapsed, 'trainer_train_runtime': float(train_result.metrics.get('train_runtime', elapsed)), 'global_step': int(trainer.state.global_step), 'best_model_checkpoint': f'in_memory_state_dict_epoch_{best_state_callback.best_epoch}', 'best_epoch': best_state_callback.best_epoch, 'best_eval_loss': best_state_callback.best_loss, 'epochs_evaluated': len(eval_entries), 'test_used_for_training_or_selection': False}
    np.savez_compressed(cache_path, dev_prob=dev_prob.astype(np.float32), test_prob=test_prob.astype(np.float32))
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    del trainer, model, tokenizer, train_dataset, dev_selection_dataset, dev_full_dataset, test_full_dataset
    if work_dir.exists():
        shutil.rmtree(work_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (dev_prob, test_prob, metadata)
