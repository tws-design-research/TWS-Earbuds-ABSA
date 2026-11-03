from __future__ import annotations
import argparse
import importlib.util
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
_COMMON_PATH = Path(__file__).with_name('absa.py')
_SPEC = importlib.util.spec_from_file_location('absa_v3_common', _COMMON_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'Cannot load common module: {_COMMON_PATH}')
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)
BASE_DIR = Path(__file__).resolve().parents[1]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--loss_mode', required=True, choices=['sample_weighted', 'aspect_balanced'])
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--pair_dir', type=Path, default=BASE_DIR / 'trained' / 'inputs')
    parser.add_argument('--base_model_dir', type=Path, default=BASE_DIR / 'pretrained' / 'distilroberta-base')
    parser.add_argument('--out_root', type=Path, default=BASE_DIR / 'trained' / 'teachers')
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=2)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lambda_positive', type=float, default=1.0)
    parser.add_argument('--lambda_negative', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--evaluate_test', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()

def autocast_context(use_amp: bool):
    if not use_amp:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)

def make_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler('cuda', enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)

def make_loader(rows: list[dict[str, Any]], tokenizer: Any, max_length: int, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    indexed = common.rows_with_indices(rows)
    dataset = common.RawPairDataset(indexed)
    collator = common.PairBatchCollator(tokenizer, max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator, num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=generator if shuffle else None, persistent_workers=bool(num_workers > 0))

def split_batch(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    labels = {'labels_mention': batch.pop('labels_mention').to(device, non_blocking=True), 'labels_positive': batch.pop('labels_positive').to(device, non_blocking=True), 'labels_negative': batch.pop('labels_negative').to(device, non_blocking=True), 'aspect_index': batch.pop('aspect_index').to(device, non_blocking=True), 'row_index': batch.pop('row_index')}
    encoder_inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return (encoder_inputs, labels)

def run_inference(model: torch.nn.Module, loader: DataLoader, n_rows: int, device: torch.device, use_amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    mention = np.zeros(n_rows, dtype=np.float32)
    positive = np.zeros(n_rows, dtype=np.float32)
    negative = np.zeros(n_rows, dtype=np.float32)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            encoder_inputs, labels = split_batch(batch, device)
            indices = labels['row_index'].cpu().numpy().astype(int)
            with autocast_context(use_amp):
                outputs = model(**encoder_inputs)
            mention[indices] = outputs['mention_logit'].float().cpu().numpy()
            positive[indices] = outputs['positive_logit'].float().cpu().numpy()
            negative[indices] = outputs['negative_logit'].float().cpu().numpy()
    return (mention, positive, negative, time.perf_counter() - start)

def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any, device: torch.device, use_amp: bool, args: argparse.Namespace, pos_weights: dict[str, float], balance_vector: torch.Tensor) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums = {'total_loss': 0.0, 'mention_loss': 0.0, 'positive_loss': 0.0, 'negative_loss': 0.0}
    n_batches = 0
    for batch_index, batch in enumerate(loader, start=1):
        encoder_inputs, labels = split_batch(batch, device)
        with autocast_context(use_amp):
            outputs = model(**encoder_inputs)
            loss, pieces = common.compute_multitask_loss(outputs=outputs, labels_mention=labels['labels_mention'], labels_positive=labels['labels_positive'], labels_negative=labels['labels_negative'], aspect_index=labels['aspect_index'], pos_weights=pos_weights, loss_mode=args.loss_mode, lambda_positive=args.lambda_positive, lambda_negative=args.lambda_negative, signal_aspect_balance_vector=balance_vector)
            scaled_loss = loss / args.gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        sums['total_loss'] += float(loss.detach().cpu())
        sums['mention_loss'] += float(pieces['mention_loss'].cpu())
        sums['positive_loss'] += float(pieces['positive_loss'].cpu())
        sums['negative_loss'] += float(pieces['negative_loss'].cpu())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in sums.items()}

def evaluate_and_save(split_name: str, rows: list[dict[str, Any]], loader: DataLoader, model: torch.nn.Module, device: torch.device, use_amp: bool, out_dir: Path, thresholds: dict[str, dict[str, float]] | None) -> tuple[dict[str, dict[str, float]], dict[str, float], float]:
    m_logit, p_logit, n_logit, inference_seconds = run_inference(model, loader, len(rows), device, use_amp)
    scores = common.pipeline_scores_from_logits(m_logit, p_logit, n_logit)
    ap_summary = common.macro_average_precision_from_scores(rows, scores)
    if thresholds is None:
        thresholds, threshold_frame = common.tune_pipeline_thresholds(rows, scores)
        threshold_frame.to_csv(out_dir / 'dev_tuned_pipeline_thresholds.csv', index=False, encoding='utf-8-sig')
    metrics, overall, groups, predictions = common.evaluate_scores(rows, scores, thresholds)
    common.save_evaluation_outputs(out_dir, split_name, rows, scores, metrics, overall, groups, predictions)
    pd.DataFrame([{'split': split_name, **ap_summary}]).to_csv(out_dir / f'{split_name}_macro_average_precision_summary.csv', index=False, encoding='utf-8-sig')
    return (thresholds, ap_summary, inference_seconds)

def main() -> None:
    args = parse_args()
    expected_loss = {'m1_shared_standard': 'sample_weighted', 'm2_shared_aspect_balanced': 'aspect_balanced'}
    if args.variant in expected_loss and expected_loss[args.variant] != args.loss_mode:
        raise ValueError(f'Variant {args.variant} must use loss_mode={expected_loss[args.variant]}, got {args.loss_mode}')
    out_dir = args.out_root / args.variant / f'seed_{args.seed:04d}'
    best_model_dir = out_dir / 'best_model'
    completed_marker = out_dir / 'PHASE1_DEV_COMPLETE.txt'
    if completed_marker.exists() and (not args.force) and (not args.eval_only):
        print(f'[SKIP] Existing completed output: {out_dir}')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    common.set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and (not args.no_amp)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    audit, loaded = common.audit_pair_files(args.pair_dir, include_test=args.evaluate_test)
    audit.to_csv(out_dir / 'input_audit.csv', index=False, encoding='utf-8-sig')
    common.fail_if_audit_failed(audit)
    train_rows = loaded['train']
    dev_rows = loaded['dev']
    test_rows = loaded.get('test', [])
    pos_weights = common.calculate_pos_weights(train_rows)
    aspect_balance_values = common.calculate_signal_aspect_balance_vector(train_rows)
    balance_vector = torch.tensor(aspect_balance_values, dtype=torch.float32, device=device)
    configuration = {'model_version': 'hierarchical_multitask_absa_v3_candidate', 'variant': args.variant, 'loss_mode': args.loss_mode, 'seed': args.seed, 'pair_dir': str(args.pair_dir), 'base_model_dir': str(args.base_model_dir), 'output_dir': str(out_dir), 'max_length': args.max_length, 'batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size, 'gradient_accumulation_steps': args.gradient_accumulation_steps, 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay, 'warmup_ratio': args.warmup_ratio, 'epochs': args.epochs, 'early_stopping_patience': args.early_stopping_patience, 'lambda_positive': args.lambda_positive, 'lambda_negative': args.lambda_negative, 'pos_weights': pos_weights, 'signal_aspect_balance_vector': aspect_balance_values, 'checkpoint_selection': 'minimum development total multitask loss', 'signal_inference': 'sigmoid(mention_logit) * sigmoid(conditional_signal_logit)', 'test_evaluation_requested': bool(args.evaluate_test), 'rating_used_as_training_target': False, 'device': str(device), 'amp': use_amp}
    common.write_json(out_dir / 'train_config.json', configuration)
    if args.eval_only:
        if not best_model_dir.exists():
            raise FileNotFoundError(f'Missing best model for eval_only: {best_model_dir}')
        tokenizer, model, _ = common.load_model_bundle(best_model_dir, device)
    else:
        tokenizer, model = common.instantiate_model_from_base(args.base_model_dir)
        model.to(device)
    train_loader = make_loader(train_rows, tokenizer, args.max_length, args.batch_size, True, args.seed, args.num_workers)
    dev_loader = make_loader(dev_rows, tokenizer, args.max_length, args.eval_batch_size, False, args.seed, args.num_workers)
    test_loader = make_loader(test_rows, tokenizer, args.max_length, args.eval_batch_size, False, args.seed, args.num_workers) if args.evaluate_test else None
    parameter_counts = common.model_parameter_counts(model)
    common.write_json(out_dir / 'parameter_counts.json', parameter_counts)
    training_seconds = 0.0
    best_epoch = None
    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        total_updates = updates_per_epoch * args.epochs
        warmup_steps = int(round(total_updates * args.warmup_ratio))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates)
        scaler = make_scaler(use_amp)
        best_dev_loss = math.inf
        patience_count = 0
        history: list[dict[str, Any]] = []
        train_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_losses = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp, args, pos_weights, balance_vector)
            m_logit, p_logit, n_logit, dev_inference_seconds = run_inference(model, dev_loader, len(dev_rows), device, use_amp)
            dev_losses = common.compute_dataset_multitask_loss(dev_rows, m_logit, p_logit, n_logit, pos_weights, args.loss_mode, args.lambda_positive, args.lambda_negative, aspect_balance_values)
            dev_scores = common.pipeline_scores_from_logits(m_logit, p_logit, n_logit)
            dev_ap = common.macro_average_precision_from_scores(dev_rows, dev_scores)
            improved = dev_losses['total_loss'] < best_dev_loss - args.min_delta
            if improved:
                best_dev_loss = dev_losses['total_loss']
                best_epoch = epoch
                patience_count = 0
                metadata = {**configuration, 'best_epoch': epoch, 'best_dev_total_loss': best_dev_loss}
                common.save_model_bundle(model, tokenizer, best_model_dir, metadata)
            else:
                patience_count += 1
            history.append({'epoch': epoch, **{f'train_{key}': value for key, value in train_losses.items()}, **{f'dev_{key}': value for key, value in dev_losses.items()}, 'dev_mention_macro_ap': dev_ap['pipeline_mention'], 'dev_positive_macro_ap': dev_ap['pipeline_positive_signal'], 'dev_negative_macro_ap': dev_ap['pipeline_negative_signal'], 'dev_mean_three_tasks_macro_ap': dev_ap['mean_three_tasks'], 'learning_rate': optimizer.param_groups[0]['lr'], 'improved': int(improved), 'patience_count': patience_count, 'epoch_seconds': time.perf_counter() - epoch_start, 'dev_inference_seconds': dev_inference_seconds})
            pd.DataFrame(history).to_csv(out_dir / 'training_history.csv', index=False, encoding='utf-8-sig')
            print(pd.DataFrame([history[-1]]).to_string(index=False))
            if patience_count >= args.early_stopping_patience:
                print(f'[EARLY STOP] epoch={epoch}; best_epoch={best_epoch}')
                break
        training_seconds = time.perf_counter() - train_start
        if best_epoch is None:
            raise RuntimeError('No best checkpoint was saved')
        tokenizer, model, _ = common.load_model_bundle(best_model_dir, device)
    thresholds, dev_ap, dev_inference_seconds = evaluate_and_save('dev', dev_rows, dev_loader, model, device, use_amp, out_dir, thresholds=None)
    test_inference_seconds = None
    if args.evaluate_test:
        if test_loader is None:
            raise RuntimeError('Internal error: test_loader was not created')
        _, _, test_inference_seconds = evaluate_and_save('test', test_rows, test_loader, model, device, use_amp, out_dir, thresholds=thresholds)
        (out_dir / 'TEST_EVALUATED.txt').write_text('Test was evaluated only because --evaluate_test was explicitly supplied.\n', encoding='utf-8')
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    efficiency = {**parameter_counts, 'best_epoch': best_epoch if best_epoch is not None else 'loaded_eval_only', 'training_seconds': training_seconds, 'dev_inference_seconds': dev_inference_seconds, 'test_inference_seconds': test_inference_seconds, 'cuda_peak_memory_bytes': peak_memory, 'dev_mean_three_tasks_macro_ap': dev_ap['mean_three_tasks']}
    pd.DataFrame([efficiency]).to_csv(out_dir / 'efficiency_metrics.csv', index=False, encoding='utf-8-sig')
    completed_marker.write_text('Phase-1 development evaluation complete. Test was not used unless TEST_EVALUATED.txt exists.\n', encoding='utf-8')
    print(f'[DONE] {args.variant} seed={args.seed}: {out_dir}')
if __name__ == '__main__':
    main()
