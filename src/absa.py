from __future__ import annotations
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
TASKS = ('y_mention', 'y_positive_signal', 'y_negative_signal')
PIPELINE_TASKS = ('pipeline_mention', 'pipeline_positive_signal', 'pipeline_negative_signal')
DEFAULT_THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
REQUIRED_PAIR_FIELDS = {'annotation_id', 'parent_asin', 'asin', 'split', 'aspect_index', 'aspect', 'aspect_level', 'include_in_core_design_priority', 'review_text_for_model', 'aspect_prompt', 'y_mention', 'y_positive_signal', 'y_negative_signal', 'y_neutral'}
EXPECTED_PAIR_ROWS = {'train': 41230, 'dev': 8722, 'test': 8162}
EXPECTED_REVIEW_ROWS = {'train': 2945, 'dev': 623, 'test': 583}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path}:{line_no}') from exc
    return rows

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out

def _clean_str(value: Any) -> str:
    return str(value if value is not None else '').strip()

def audit_pair_files(pair_dir: Path, include_test: bool=True) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    splits = ['train', 'dev'] + (['test'] if include_test else [])
    loaded: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []

    def add(check: str, value: Any, status: str, detail: str='') -> None:
        audit.append({'check': check, 'value': value, 'status': status, 'detail': detail})
    for split in splits:
        path = pair_dir / f'{split}_aspect_pairs.jsonl'
        add(f'{split}_file_exists', str(path), 'PASS' if path.exists() else 'FAIL')
        if not path.exists():
            continue
        rows = read_jsonl(path)
        loaded[split] = rows
        add(f'{split}_pair_rows', len(rows), 'PASS' if len(rows) == EXPECTED_PAIR_ROWS[split] else 'FAIL', f'expected={EXPECTED_PAIR_ROWS[split]}')
        if not rows:
            add(f'{split}_nonempty', 0, 'FAIL')
            continue
        missing_fields = sorted(REQUIRED_PAIR_FIELDS - set(rows[0]))
        add(f'{split}_required_fields', ';'.join(missing_fields), 'PASS' if not missing_fields else 'FAIL')
        frame = pd.DataFrame(rows)
        n_reviews = int(frame['annotation_id'].nunique())
        n_aspects = int(frame['aspect'].nunique())
        add(f'{split}_review_rows', n_reviews, 'PASS' if n_reviews == EXPECTED_REVIEW_ROWS[split] else 'FAIL', f'expected={EXPECTED_REVIEW_ROWS[split]}')
        add(f'{split}_n_aspects', n_aspects, 'PASS' if n_aspects == 14 else 'FAIL')
        per_review = frame.groupby('annotation_id')['aspect'].agg(['size', 'nunique'])
        complete = bool(((per_review['size'] == 14) & (per_review['nunique'] == 14)).all())
        add(f'{split}_complete_14_aspects_per_review', complete, 'PASS' if complete else 'FAIL')
        duplicate_pairs = int(frame.duplicated(['annotation_id', 'aspect']).sum())
        add(f'{split}_duplicate_annotation_aspect', duplicate_pairs, 'PASS' if duplicate_pairs == 0 else 'FAIL')
        binary_ok = True
        for col in ('y_mention', 'y_positive_signal', 'y_negative_signal', 'y_neutral'):
            values = set(pd.to_numeric(frame[col], errors='coerce').dropna().astype(int).tolist())
            if not values.issubset({0, 1}):
                binary_ok = False
        add(f'{split}_binary_labels', binary_ok, 'PASS' if binary_ok else 'FAIL')
        y_m = frame['y_mention'].astype(int)
        y_p = frame['y_positive_signal'].astype(int)
        y_n = frame['y_negative_signal'].astype(int)
        y_z = frame['y_neutral'].astype(int)
        logical_violations = int(((y_p > y_m) | (y_n > y_m) | (y_z > y_m) | (y_z == 1) & ((y_p == 1) | (y_n == 1))).sum())
        add(f'{split}_label_logic_violations', logical_violations, 'PASS' if logical_violations == 0 else 'FAIL')
        split_values = sorted({_clean_str(x) for x in frame['split'].unique()})
        add(f'{split}_split_field', ';'.join(split_values), 'PASS' if split_values == [split] else 'FAIL')
        prompt_values = sorted({_clean_str(x) for x in frame.get('prompt_variant', pd.Series(dtype=str)).unique()})
        add(f'{split}_prompt_variant', ';'.join(prompt_values), 'INFO')
        aspect_index_check = frame.groupby('aspect')['aspect_index'].nunique().max() == 1
        add(f'{split}_stable_aspect_index', bool(aspect_index_check), 'PASS' if aspect_index_check else 'FAIL')
        empty_review_count = int(frame.groupby('annotation_id')['empty_review_text_flag'].max().sum()) if 'empty_review_text_flag' in frame else 0
        add(f'{split}_empty_review_count', empty_review_count, 'INFO')
    for key in ('annotation_id', 'parent_asin', 'asin'):
        for left, right in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
            if left not in loaded or right not in loaded:
                continue
            left_values = {_clean_str(r.get(key)) for r in loaded[left] if _clean_str(r.get(key))}
            right_values = {_clean_str(r.get(key)) for r in loaded[right] if _clean_str(r.get(key))}
            overlap = len(left_values & right_values)
            add(f'overlap_{key}_{left}_{right}', overlap, 'PASS' if overlap == 0 else 'FAIL')
    audit_df = pd.DataFrame(audit)
    return (audit_df, loaded)

def fail_if_audit_failed(audit_df: pd.DataFrame) -> None:
    failed = audit_df[audit_df['status'] == 'FAIL']
    if not failed.empty:
        raise RuntimeError('Input audit failed:\n' + failed.to_string(index=False))

def infer_aspect_metadata(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    metadata = pd.DataFrame([{'aspect': r['aspect'], 'aspect_index': int(r['aspect_index']), 'aspect_level': _clean_str(r.get('aspect_level')), 'include_in_core_design_priority': int(str(r.get('include_in_core_design_priority', '0')) or 0)} for r in rows]).drop_duplicates('aspect')
    if len(metadata) != 14:
        raise ValueError(f'Expected 14 unique aspects, found {len(metadata)}')
    return metadata.sort_values('aspect_index').reset_index(drop=True)

def calculate_pos_weights(train_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    frame = pd.DataFrame(train_rows)
    weights: dict[str, float] = {}
    mention_pos = int(frame['y_mention'].sum())
    mention_neg = len(frame) - mention_pos
    weights['y_mention'] = mention_neg / max(mention_pos, 1)
    mentioned = frame[frame['y_mention'].astype(int) == 1]
    for task in ('y_positive_signal', 'y_negative_signal'):
        n_pos = int(mentioned[task].sum())
        n_neg = len(mentioned) - n_pos
        weights[task] = n_neg / max(n_pos, 1)
    return weights

def calculate_signal_aspect_balance_vector(train_rows: Sequence[dict[str, Any]]) -> list[float]:
    frame = pd.DataFrame(train_rows)
    mentioned = frame[frame['y_mention'].astype(int) == 1]
    counts = mentioned.groupby('aspect_index').size().to_dict()
    n_aspects = int(frame['aspect_index'].nunique())
    total_active = int(len(mentioned))
    vector: list[float] = []
    for aspect_index in range(n_aspects):
        count = int(counts.get(aspect_index, 0))
        if count <= 0:
            raise ValueError(f'No mentioned training examples for aspect_index={aspect_index}')
        vector.append(total_active / (n_aspects * count))
    return vector

class RawPairDataset(Dataset):

    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

class PairBatchCollator:

    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer([str(r['review_text_for_model']) for r in batch], [str(r['aspect_prompt']) for r in batch], truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        encoded['labels_mention'] = torch.tensor([int(r['y_mention']) for r in batch], dtype=torch.float32)
        encoded['labels_positive'] = torch.tensor([int(r['y_positive_signal']) for r in batch], dtype=torch.float32)
        encoded['labels_negative'] = torch.tensor([int(r['y_negative_signal']) for r in batch], dtype=torch.float32)
        encoded['aspect_index'] = torch.tensor([int(r['aspect_index']) for r in batch], dtype=torch.long)
        encoded['row_index'] = torch.tensor([int(r['__row_index__']) for r in batch], dtype=torch.long)
        return encoded

class RobertaStyleBinaryHead(nn.Module):

    def __init__(self, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x).squeeze(-1)

class HierarchicalMultiTaskABSA(nn.Module):

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        config = encoder.config
        hidden_size = int(config.hidden_size)
        dropout_prob = getattr(config, 'classifier_dropout', None)
        if dropout_prob is None:
            dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        self.mention_head = RobertaStyleBinaryHead(hidden_size, float(dropout_prob))
        self.positive_head = RobertaStyleBinaryHead(hidden_size, float(dropout_prob))
        self.negative_head = RobertaStyleBinaryHead(hidden_size, float(dropout_prob))

    def forward(self, **encoder_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(**encoder_inputs)
        cls_state = outputs.last_hidden_state[:, 0, :]
        return {'mention_logit': self.mention_head(cls_state), 'positive_logit': self.positive_head(cls_state), 'negative_logit': self.negative_head(cls_state)}

def instantiate_model_from_base(base_model_dir: Path) -> tuple[Any, HierarchicalMultiTaskABSA]:
    if not base_model_dir.exists():
        raise FileNotFoundError(f'Missing local base model: {base_model_dir}')
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(base_model_dir), local_files_only=True)
    return (tokenizer, HierarchicalMultiTaskABSA(encoder))

def save_model_bundle(model: HierarchicalMultiTaskABSA, tokenizer: Any, output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = output_dir / 'encoder'
    model.encoder.save_pretrained(str(encoder_dir))
    tokenizer.save_pretrained(str(output_dir))
    torch.save({'mention_head': model.mention_head.state_dict(), 'positive_head': model.positive_head.state_dict(), 'negative_head': model.negative_head.state_dict()}, output_dir / 'task_heads.pt')
    write_json(output_dir / 'model_bundle_config.json', metadata)

def load_model_bundle(model_dir: Path, device: torch.device) -> tuple[Any, HierarchicalMultiTaskABSA, dict[str, Any]]:
    config_path = model_dir / 'model_bundle_config.json'
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    metadata = json.loads(config_path.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(model_dir / 'encoder'), local_files_only=True)
    model = HierarchicalMultiTaskABSA(encoder)
    try:
        heads = torch.load(model_dir / 'task_heads.pt', map_location='cpu', weights_only=True)
    except TypeError:
        heads = torch.load(model_dir / 'task_heads.pt', map_location='cpu')
    model.mention_head.load_state_dict(heads['mention_head'])
    model.positive_head.load_state_dict(heads['positive_head'])
    model.negative_head.load_state_dict(heads['negative_head'])
    model.to(device)
    return (tokenizer, model, metadata)

def _weighted_bce_per_example(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float) -> torch.Tensor:
    weight = torch.tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction='none')

def masked_loss(logits: torch.Tensor, labels: torch.Tensor, active_mask: torch.Tensor, aspect_index: torch.Tensor, pos_weight: float, loss_mode: str, signal_aspect_balance_vector: torch.Tensor | None=None) -> torch.Tensor:
    per_example = _weighted_bce_per_example(logits, labels, pos_weight)
    active_mask = active_mask.bool()
    if not bool(active_mask.any()):
        return logits.sum() * 0.0
    if loss_mode == 'sample_weighted':
        return per_example[active_mask].mean()
    if loss_mode != 'aspect_balanced':
        raise ValueError(f'Unsupported loss_mode: {loss_mode}')
    if signal_aspect_balance_vector is None:
        raise ValueError('aspect_balanced loss requires signal_aspect_balance_vector')
    example_weights = signal_aspect_balance_vector[aspect_index]
    return (per_example[active_mask] * example_weights[active_mask]).mean()

def compute_multitask_loss(outputs: dict[str, torch.Tensor], labels_mention: torch.Tensor, labels_positive: torch.Tensor, labels_negative: torch.Tensor, aspect_index: torch.Tensor, pos_weights: dict[str, float], loss_mode: str, lambda_positive: float, lambda_negative: float, signal_aspect_balance_vector: torch.Tensor | None=None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mention_active = torch.ones_like(labels_mention, dtype=torch.bool)
    signal_active = labels_mention > 0.5
    mention_loss = masked_loss(outputs['mention_logit'], labels_mention, mention_active, aspect_index, pos_weights['y_mention'], 'sample_weighted', None)
    positive_loss = masked_loss(outputs['positive_logit'], labels_positive, signal_active, aspect_index, pos_weights['y_positive_signal'], loss_mode, signal_aspect_balance_vector)
    negative_loss = masked_loss(outputs['negative_logit'], labels_negative, signal_active, aspect_index, pos_weights['y_negative_signal'], loss_mode, signal_aspect_balance_vector)
    total = mention_loss + float(lambda_positive) * positive_loss + float(lambda_negative) * negative_loss
    return (total, {'mention_loss': mention_loss.detach(), 'positive_loss': positive_loss.detach(), 'negative_loss': negative_loss.detach()})

def compute_dataset_multitask_loss(rows: Sequence[dict[str, Any]], mention_logits: np.ndarray, positive_logits: np.ndarray, negative_logits: np.ndarray, pos_weights: dict[str, float], loss_mode: str, lambda_positive: float, lambda_negative: float, signal_aspect_balance_vector: Sequence[float]) -> dict[str, float]:
    labels_m = torch.tensor([int(r['y_mention']) for r in rows], dtype=torch.float32)
    labels_p = torch.tensor([int(r['y_positive_signal']) for r in rows], dtype=torch.float32)
    labels_n = torch.tensor([int(r['y_negative_signal']) for r in rows], dtype=torch.float32)
    aspects = torch.tensor([int(r['aspect_index']) for r in rows], dtype=torch.long)
    logits = {'mention_logit': torch.tensor(mention_logits, dtype=torch.float32), 'positive_logit': torch.tensor(positive_logits, dtype=torch.float32), 'negative_logit': torch.tensor(negative_logits, dtype=torch.float32)}
    balance = torch.tensor(list(signal_aspect_balance_vector), dtype=torch.float32)
    total, pieces = compute_multitask_loss(logits, labels_m, labels_p, labels_n, aspects, pos_weights, loss_mode, lambda_positive, lambda_negative, balance)
    return {'total_loss': float(total.item()), 'mention_loss': float(pieces['mention_loss'].item()), 'positive_loss': float(pieces['positive_loss'].item()), 'negative_loss': float(pieces['negative_loss'].item())}

def rows_with_indices(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        copied = dict(row)
        copied['__row_index__'] = index
        output.append(copied)
    return output

def pipeline_scores_from_logits(mention_logits: np.ndarray, positive_logits: np.ndarray, negative_logits: np.ndarray) -> dict[str, np.ndarray]:
    mention_score = sigmoid_numpy(mention_logits)
    positive_conditional_score = sigmoid_numpy(positive_logits)
    negative_conditional_score = sigmoid_numpy(negative_logits)
    return {'mention_score': mention_score, 'positive_conditional_score': positive_conditional_score, 'negative_conditional_score': negative_conditional_score, 'positive_pipeline_score': mention_score * positive_conditional_score, 'negative_pipeline_score': mention_score * negative_conditional_score}

def tune_thresholds_by_aspect(rows: Sequence[dict[str, Any]], y_true: np.ndarray, scores: np.ndarray, threshold_grid: np.ndarray=DEFAULT_THRESHOLD_GRID) -> tuple[dict[str, float], pd.DataFrame]:
    by_aspect: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_aspect[str(row['aspect'])].append(index)
    thresholds: dict[str, float] = {}
    selected: list[dict[str, Any]] = []
    for aspect, indices_list in sorted(by_aspect.items()):
        indices = np.asarray(indices_list, dtype=int)
        target = y_true[indices]
        aspect_scores = scores[indices]
        best: dict[str, Any] | None = None
        for threshold in threshold_grid:
            pred = (aspect_scores >= float(threshold)).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary', zero_division=0)
            candidate = {'aspect': aspect, 'threshold': float(threshold), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'support_positive': int(target.sum()), 'support_total': int(len(target)), 'predicted_positive': int(pred.sum())}
            key = (candidate['f1'], candidate['precision'], candidate['recall'], candidate['threshold'])
            if best is None:
                best = candidate
            else:
                best_key = (best['f1'], best['precision'], best['recall'], best['threshold'])
                if key > best_key:
                    best = candidate
        assert best is not None
        thresholds[aspect] = float(best['threshold'])
        selected.append(best)
    return (thresholds, pd.DataFrame(selected))

def apply_thresholds(rows: Sequence[dict[str, Any]], scores: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    output = np.zeros(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        threshold = float(thresholds.get(str(row['aspect']), 0.5))
        output[index] = int(scores[index] >= threshold)
    return output

def metrics_by_aspect(rows: Sequence[dict[str, Any]], y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, task_name: str) -> pd.DataFrame:
    metadata = infer_aspect_metadata(rows)
    metadata_by_aspect = metadata.set_index('aspect').to_dict(orient='index')
    by_aspect: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_aspect[str(row['aspect'])].append(index)
    output: list[dict[str, Any]] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    for aspect, index_list in sorted(by_aspect.items()):
        indices = np.asarray(index_list, dtype=int)
        target = y_true[indices]
        pred = y_pred[indices]
        precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary', zero_division=0)
        average_precision = average_precision_score(target, scores[indices]) if len(np.unique(target)) > 1 else np.nan
        meta = metadata_by_aspect[aspect]
        output.append({'task': task_name, 'aspect': aspect, 'aspect_index': int(meta['aspect_index']), 'aspect_level': meta['aspect_level'], 'include_in_core_design_priority': int(meta['include_in_core_design_priority']), 'support_positive': int(target.sum()), 'support_total': int(len(target)), 'predicted_positive': int(pred.sum()), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'average_precision': float(average_precision) if not np.isnan(average_precision) else np.nan})
        all_true.extend(target.tolist())
        all_pred.extend(pred.tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary', zero_division=0)
    output.append({'task': task_name, 'aspect': '__micro_all_pairs__', 'aspect_index': -1, 'aspect_level': 'all_pairs', 'include_in_core_design_priority': -1, 'support_positive': int(sum(all_true)), 'support_total': int(len(all_true)), 'predicted_positive': int(sum(all_pred)), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'average_precision': np.nan})
    return pd.DataFrame(output)

def overall_summary(metric_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for task, frame in metric_frames.items():
        aspects = frame[frame['aspect'] != '__micro_all_pairs__'].copy()
        micro = frame[frame['aspect'] == '__micro_all_pairs__'].iloc[0]
        output.append({'task': task, 'macro_precision': float(aspects['precision'].mean()), 'macro_recall': float(aspects['recall'].mean()), 'macro_f1': float(aspects['f1'].mean()), 'macro_average_precision': float(aspects['average_precision'].mean()), 'micro_precision': float(micro['precision']), 'micro_recall': float(micro['recall']), 'micro_f1': float(micro['f1']), 'support_positive': int(aspects['support_positive'].sum()), 'support_total': int(aspects['support_total'].sum()), 'predicted_positive': int(aspects['predicted_positive'].sum())})
    return pd.DataFrame(output)

def group_summary(metric_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for task, frame in metric_frames.items():
        aspects = frame[frame['aspect'] != '__micro_all_pairs__'].copy()
        group_masks = {'all_14_aspects': pd.Series(True, index=aspects.index), 'core_design_10': aspects['include_in_core_design_priority'].astype(int) == 1, 'contextual_product_experience': aspects['aspect_level'].eq('contextual_product_experience'), 'controls': aspects['aspect_level'].isin(['market_control', 'non_design_control'])}
        for group_name, mask in group_masks.items():
            subset = aspects[mask]
            if subset.empty:
                continue
            output.append({'task': task, 'aspect_group': group_name, 'n_aspects': int(len(subset)), 'macro_precision': float(subset['precision'].mean()), 'macro_recall': float(subset['recall'].mean()), 'macro_f1': float(subset['f1'].mean()), 'macro_average_precision': float(subset['average_precision'].mean()), 'support_positive': int(subset['support_positive'].sum()), 'support_total': int(subset['support_total'].sum())})
    return pd.DataFrame(output)

def evaluate_scores(rows: Sequence[dict[str, Any]], scores: dict[str, np.ndarray], thresholds_by_task: dict[str, dict[str, float]]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    y_mention = np.asarray([int(r['y_mention']) for r in rows], dtype=int)
    y_positive = np.asarray([int(r['y_positive_signal']) for r in rows], dtype=int)
    y_negative = np.asarray([int(r['y_negative_signal']) for r in rows], dtype=int)
    pred_mention = apply_thresholds(rows, scores['mention_score'], thresholds_by_task['pipeline_mention'])
    pred_positive = apply_thresholds(rows, scores['positive_pipeline_score'], thresholds_by_task['pipeline_positive_signal'])
    pred_negative = apply_thresholds(rows, scores['negative_pipeline_score'], thresholds_by_task['pipeline_negative_signal'])
    metrics = {'pipeline_mention': metrics_by_aspect(rows, y_mention, pred_mention, scores['mention_score'], 'pipeline_mention'), 'pipeline_positive_signal': metrics_by_aspect(rows, y_positive, pred_positive, scores['positive_pipeline_score'], 'pipeline_positive_signal'), 'pipeline_negative_signal': metrics_by_aspect(rows, y_negative, pred_negative, scores['negative_pipeline_score'], 'pipeline_negative_signal')}
    predictions = {'pred_mention': pred_mention, 'pred_positive_signal': pred_positive, 'pred_negative_signal': pred_negative}
    return (metrics, overall_summary(metrics), group_summary(metrics), predictions)

def tune_pipeline_thresholds(rows: Sequence[dict[str, Any]], scores: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    targets_and_scores = {'pipeline_mention': (np.asarray([int(r['y_mention']) for r in rows], dtype=int), scores['mention_score']), 'pipeline_positive_signal': (np.asarray([int(r['y_positive_signal']) for r in rows], dtype=int), scores['positive_pipeline_score']), 'pipeline_negative_signal': (np.asarray([int(r['y_negative_signal']) for r in rows], dtype=int), scores['negative_pipeline_score'])}
    threshold_maps: dict[str, dict[str, float]] = {}
    frames: list[pd.DataFrame] = []
    for task, (target, task_scores) in targets_and_scores.items():
        mapping, frame = tune_thresholds_by_aspect(rows, target, task_scores)
        frame.insert(0, 'task', task)
        threshold_maps[task] = mapping
        frames.append(frame)
    return (threshold_maps, pd.concat(frames, ignore_index=True))

def save_evaluation_outputs(out_dir: Path, split_name: str, rows: Sequence[dict[str, Any]], scores: dict[str, np.ndarray], metrics: dict[str, pd.DataFrame], overall: pd.DataFrame, groups: pd.DataFrame, predictions: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, frame in metrics.items():
        frame.to_csv(out_dir / f'{split_name}_{task}_metrics_by_aspect.csv', index=False, encoding='utf-8-sig')
    pd.concat(metrics.values(), ignore_index=True).to_csv(out_dir / f'{split_name}_pipeline_all_metrics_by_aspect.csv', index=False, encoding='utf-8-sig')
    overall.to_csv(out_dir / f'{split_name}_pipeline_overall_summary.csv', index=False, encoding='utf-8-sig')
    groups.to_csv(out_dir / f'{split_name}_pipeline_group_summary.csv', index=False, encoding='utf-8-sig')
    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prediction_rows.append({'annotation_id': row.get('annotation_id', ''), 'parent_asin': row.get('parent_asin', ''), 'asin': row.get('asin', ''), 'aspect': row.get('aspect', ''), 'aspect_index': row.get('aspect_index', ''), 'aspect_level': row.get('aspect_level', ''), 'include_in_core_design_priority': row.get('include_in_core_design_priority', ''), 'rating': row.get('rating', ''), 'raw_label': row.get('raw_label', ''), 'y_mention_true': int(row['y_mention']), 'y_positive_signal_true': int(row['y_positive_signal']), 'y_negative_signal_true': int(row['y_negative_signal']), 'score_mention': float(scores['mention_score'][index]), 'score_positive_conditional': float(scores['positive_conditional_score'][index]), 'score_negative_conditional': float(scores['negative_conditional_score'][index]), 'score_positive_pipeline': float(scores['positive_pipeline_score'][index]), 'score_negative_pipeline': float(scores['negative_pipeline_score'][index]), 'pred_mention': int(predictions['pred_mention'][index]), 'pred_positive_signal': int(predictions['pred_positive_signal'][index]), 'pred_negative_signal': int(predictions['pred_negative_signal'][index]), 'review_text_for_model': row.get('review_text_for_model', ''), 'aspect_prompt': row.get('aspect_prompt', '')})
    pd.DataFrame(prediction_rows).to_csv(out_dir / f'{split_name}_pipeline_predictions.csv', index=False, encoding='utf-8-sig')

def macro_average_precision_from_scores(rows: Sequence[dict[str, Any]], scores: dict[str, np.ndarray]) -> dict[str, float]:
    mapping = {'pipeline_mention': ('y_mention', 'mention_score'), 'pipeline_positive_signal': ('y_positive_signal', 'positive_pipeline_score'), 'pipeline_negative_signal': ('y_negative_signal', 'negative_pipeline_score')}
    result: dict[str, float] = {}
    for task, (label_key, score_key) in mapping.items():
        by_aspect: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_aspect[str(row['aspect'])].append(index)
        values: list[float] = []
        for indices_list in by_aspect.values():
            indices = np.asarray(indices_list, dtype=int)
            target = np.asarray([int(rows[i][label_key]) for i in indices], dtype=int)
            if len(np.unique(target)) < 2:
                continue
            values.append(float(average_precision_score(target, scores[score_key][indices])))
        result[task] = float(np.mean(values)) if values else float('nan')
    result['mean_three_tasks'] = float(np.nanmean(list(result.values())))
    return result

def model_parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum((parameter.numel() for parameter in model.parameters()))
    trainable = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    encoder = sum((parameter.numel() for parameter in model.encoder.parameters()))
    heads = total - encoder
    return {'total_parameters': int(total), 'trainable_parameters': int(trainable), 'encoder_parameters': int(encoder), 'task_head_parameters': int(heads)}
