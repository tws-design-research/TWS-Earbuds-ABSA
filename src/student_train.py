from __future__ import annotations
import argparse
import importlib.util
import json
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
SCRIPT_DIR = Path(__file__).resolve().parent

def load_module(filename: str, module_name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
repair = load_module('student.py', 'dcq_repair_common')
project = load_module('absa.py', 'absa_v3_common')
BASE_DIR = Path(__file__).resolve().parents[1]
VARIANT_CONFIG = {'student': {'use_codebook_gate': False, 'lambda_relation_kd': 0.3, 'lambda_boundary': 0.0}}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=sorted(VARIANT_CONFIG), required=True)
    parser.add_argument('--encoder_source', choices=['generic', 'm2'], required=True)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--data_dir', type=Path, default=BASE_DIR / 'trained' / 'inputs')
    parser.add_argument('--base_model_dir', type=Path, default=BASE_DIR / 'pretrained' / 'distilroberta-base')
    parser.add_argument('--m2_model_dir', type=Path, default=BASE_DIR / 'trained' / 'teachers' / 'teacher' / 'seed_0020' / 'best_model')
    parser.add_argument('--codebook', type=Path, default=BASE_DIR / 'data' / 'codebook.csv')
    parser.add_argument('--teacher_dir', type=Path, default=BASE_DIR / 'trained' / 'teacher_logits' / 'seed_0020')
    parser.add_argument('--boundary_graph', type=Path, default=None)
    parser.add_argument('--out_root', type=Path, default=BASE_DIR / 'trained' / 'students')
    parser.add_argument('--interaction_rank', type=int, default=192)
    parser.add_argument('--initial_gate_probability', type=float, default=0.01)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--encoder_learning_rate', type=float, default=1e-05)
    parser.add_argument('--module_learning_rate', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--boundary_margin', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_amp', action='store_true')
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

def make_loader(rows, tokenizer, max_length, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(repair.ReviewDataset(rows), batch_size=batch_size, shuffle=shuffle, collate_fn=repair.ReviewBatchCollator(tokenizer, max_length), num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=generator if shuffle else None, persistent_workers=bool(num_workers > 0))

def split_batch(batch: dict[str, Any], device: torch.device):
    labels = {'labels_mention': batch.pop('labels_mention').to(device, non_blocking=True), 'labels_positive': batch.pop('labels_positive').to(device, non_blocking=True), 'labels_negative': batch.pop('labels_negative').to(device, non_blocking=True), 'row_index': batch.pop('row_index')}
    inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return (inputs, labels)

def load_teacher_matrix(path: Path, rows: list[dict[str, Any]], expected_aspects: list[str]) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    ids = data['annotation_ids'].astype(str).tolist()
    aspects = data['aspects'].astype(str).tolist()
    logits = np.asarray(data['logits'], dtype=np.float32)
    if aspects != expected_aspects:
        raise RuntimeError(f'Teacher aspect order mismatch. teacher={aspects}; data={expected_aspects}')
    if logits.shape[1:] != (14, 3):
        raise RuntimeError(f'Invalid teacher shape: {logits.shape}')
    if len(ids) != len(set(ids)):
        raise RuntimeError('Duplicate annotation IDs in teacher logit file')
    lookup = {annotation_id: index for index, annotation_id in enumerate(ids)}
    missing = [str(row['annotation_id']) for row in rows if str(row['annotation_id']) not in lookup]
    if missing:
        raise RuntimeError(f'Teacher logits missing {len(missing)} review IDs; first={missing[:3]}')
    aligned = np.stack([logits[lookup[str(row['annotation_id'])]] for row in rows])
    return torch.tensor(aligned, dtype=torch.float32)

def load_primary_edges(path: Path, aspects: list[str]) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding='utf-8-sig')
    if 'primary_training_edge' in frame.columns:
        frame = frame[frame['primary_training_edge'].astype(int) == 1]
    index = {aspect: i for i, aspect in enumerate(aspects)}
    edges: list[tuple[int, int]] = []
    for row in frame.itertuples(index=False):
        left, right = (str(row.aspect_a), str(row.aspect_b))
        if left not in index or right not in index:
            raise RuntimeError(f'Unknown graph edge: {left}, {right}')
        edge = (index[left], index[right])
        if edge not in edges and (edge[1], edge[0]) not in edges:
            edges.append(edge)
    if not edges:
        raise RuntimeError('Boundary variant requires at least one primary edge')
    return edges

def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = bool(trainable)

def parameter_groups(model: torch.nn.Module, encoder_lr: float, module_lr: float, weight_decay: float):
    no_decay_tokens = ('bias', 'LayerNorm.weight', 'layer_norm.weight', 'query_norm.weight')
    grouped: list[dict[str, Any]] = []
    buckets: dict[tuple[str, bool], list[torch.nn.Parameter]] = {('encoder', True): [], ('encoder', False): [], ('module', True): [], ('module', False): []}
    for name, parameter in model.named_parameters():
        source = 'encoder' if name.startswith('encoder.') else 'module'
        use_decay = not any((token in name for token in no_decay_tokens))
        buckets[source, use_decay].append(parameter)
    for (source, use_decay), parameters in buckets.items():
        if not parameters:
            continue
        grouped.append({'params': parameters, 'lr': float(encoder_lr if source == 'encoder' else module_lr), 'weight_decay': float(weight_decay if use_decay else 0.0), 'group_name': f"{source}_{('decay' if use_decay else 'no_decay')}"})
    return grouped

def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        norm = float(parameter.grad.detach().float().norm(2).cpu())
        total += norm * norm
    return math.sqrt(total) if found else 0.0

def compute_dev_gold_loss(model, loader, device, use_amp, pos_weights, aspect_balance):
    model.eval()
    sums = {'gold_total_loss': 0.0, 'mention_loss': 0.0, 'positive_loss': 0.0, 'negative_loss': 0.0}
    n_reviews = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = split_batch(batch, device)
            with autocast_context(use_amp):
                outputs = model(**inputs)
                total, pieces = repair.gold_multitask_loss(outputs, labels['labels_mention'], labels['labels_positive'], labels['labels_negative'], pos_weights, aspect_balance)
            batch_n = int(labels['labels_mention'].size(0))
            sums['gold_total_loss'] += float(total.detach().cpu()) * batch_n
            for key in ('mention_loss', 'positive_loss', 'negative_loss'):
                sums[key] += float(pieces[key].cpu()) * batch_n
            n_reviews += batch_n
    return {key: value / max(n_reviews, 1) for key, value in sums.items()}

def run_inference(model, loader, n_rows, device, use_amp):
    model.eval()
    mention = np.zeros((n_rows, 14), dtype=np.float32)
    positive = np.zeros((n_rows, 14), dtype=np.float32)
    negative = np.zeros((n_rows, 14), dtype=np.float32)
    entropy_sum = np.zeros(14, dtype=np.float64)
    top1_sum = np.zeros(14, dtype=np.float64)
    n_entropy = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            inputs, labels = split_batch(batch, device)
            indices = labels['row_index'].cpu().numpy().astype(int)
            with autocast_context(use_amp):
                outputs = model(**inputs)
            mention[indices] = outputs['mention_logit'].float().cpu().numpy()
            positive[indices] = outputs['positive_logit'].float().cpu().numpy()
            negative[indices] = outputs['negative_logit'].float().cpu().numpy()
            attention = outputs['attention'].float()
            mask = inputs.get('attention_mask')
            if mask is None:
                valid_len = torch.full((attention.size(0),), attention.size(-1), device=attention.device, dtype=torch.float32)
            else:
                valid_len = mask.float().sum(dim=-1).clamp_min(2.0)
            entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(dim=-1)
            normalized = entropy / valid_len.log().unsqueeze(1)
            entropy_sum += normalized.sum(dim=0).cpu().numpy()
            top1_sum += attention.max(dim=-1).values.sum(dim=0).cpu().numpy()
            n_entropy += int(attention.size(0))
    diagnostics = {'normalized_attention_entropy': entropy_sum / max(n_entropy, 1), 'attention_top1_mass': top1_sum / max(n_entropy, 1)}
    return (mention, positive, negative, diagnostics, time.perf_counter() - start)

def evaluate_dev(model, tokenizer, dev_rows, args, device, use_amp, out_dir):
    loader = make_loader(dev_rows, tokenizer, args.max_length, args.eval_batch_size, False, args.seed, args.num_workers)
    m, p, n, attention_diag, seconds = run_inference(model, loader, len(dev_rows), device, use_amp)
    review_scores = repair.pipeline_scores_from_logits(m, p, n)
    pair_rows = repair.flatten_review_rows(dev_rows)
    flat_scores = repair.flatten_scores(review_scores)
    thresholds, threshold_frame = project.tune_pipeline_thresholds(pair_rows, flat_scores)
    metrics, overall, groups, predictions = project.evaluate_scores(pair_rows, flat_scores, thresholds)
    project.save_evaluation_outputs(out_dir, 'dev', pair_rows, flat_scores, metrics, overall, groups, predictions)
    threshold_frame.to_csv(out_dir / 'dev_tuned_pipeline_thresholds.csv', index=False, encoding='utf-8-sig')
    ap = project.macro_average_precision_from_scores(pair_rows, flat_scores)
    pd.DataFrame([{'split': 'dev', **ap}]).to_csv(out_dir / 'dev_macro_average_precision_summary.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(out_dir / 'dev_logits.npz', annotation_ids=np.asarray([str(row['annotation_id']) for row in dev_rows], dtype=str), aspects=np.asarray(dev_rows[0]['aspects'], dtype=str), logits=np.stack([m, p, n], axis=-1))
    query_frame, query_summary = repair.query_diagnostics(model, dev_rows[0]['aspects'])
    query_frame['normalized_attention_entropy'] = attention_diag['normalized_attention_entropy']
    query_frame['attention_top1_mass'] = attention_diag['attention_top1_mass']
    query_frame.to_csv(out_dir / 'dev_aspect_query_diagnostics.csv', index=False, encoding='utf-8-sig')
    repair.write_json(out_dir / 'dev_query_diagnostics_summary.json', query_summary)
    return (ap, seconds, query_summary)

def main() -> None:
    args = parse_args()
    config = VARIANT_CONFIG[args.variant]
    run_name = f'{args.variant}__encoder_{args.encoder_source}'
    out_dir = args.out_root / run_name / f'seed_{args.seed:04d}'
    marker = out_dir / 'ROUND2_DEV_COMPLETE.txt'
    if marker.exists() and (not args.force):
        print(f'[SKIP] Completed: {out_dir}')
        return
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repair.set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and (not args.no_amp)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    train_rows = repair.read_jsonl(args.data_dir / 'train_reviews_14aspect.jsonl')
    dev_rows = repair.read_jsonl(args.data_dir / 'dev_reviews_14aspect.jsonl')
    if len(train_rows) != 2945 or len(dev_rows) != 623:
        raise RuntimeError(f'Unexpected review counts: train={len(train_rows)}, dev={len(dev_rows)}')
    aspects = [str(value) for value in train_rows[0]['aspects']]
    if len(aspects) != 14 or [str(v) for v in dev_rows[0]['aspects']] != aspects:
        raise RuntimeError('Train/dev aspect order mismatch')
    codebook = pd.read_csv(args.codebook, encoding='utf-8-sig')
    codebook['aspect_index'] = codebook['aspect'].map({aspect: i for i, aspect in enumerate(aspects)})
    if codebook['aspect_index'].isna().any() or len(codebook) != 14:
        raise RuntimeError('Codebook aspects do not match review-level data')
    codebook = codebook.sort_values('aspect_index').reset_index(drop=True)
    encoder_dir = args.base_model_dir if args.encoder_source == 'generic' else args.m2_model_dir / 'encoder'
    tokenizer, model = repair.instantiate_repair_student(tokenizer_dir=args.base_model_dir, encoder_dir=encoder_dir, codebook=codebook, device=device, interaction_rank=args.interaction_rank, use_codebook_gate=config['use_codebook_gate'], initial_gate_probability=args.initial_gate_probability)
    train_loader = make_loader(train_rows, tokenizer, args.max_length, args.batch_size, True, args.seed, args.num_workers)
    dev_loader = make_loader(dev_rows, tokenizer, args.max_length, args.eval_batch_size, False, args.seed, args.num_workers)
    pos_weights = repair.calculate_pos_weights(train_rows)
    aspect_balance = repair.calculate_aspect_balance(train_rows).to(device)
    teacher_train = None
    if config['lambda_relation_kd'] > 0:
        teacher_train = load_teacher_matrix(args.teacher_dir / 'teacher_train_logits.npz', train_rows, aspects)
    primary_edges: list[tuple[int, int]] = []
    if config['lambda_boundary'] > 0:
        primary_edges = load_primary_edges(args.boundary_graph, aspects)
    run_config = {'model': 'TWS Student', 'run_name': run_name, 'variant': args.variant, 'encoder_source': args.encoder_source, 'encoder_dir': str(encoder_dir), 'seed': args.seed, 'single_pass_review_encoder': True, 'n_aspect_queries': 14, 'interaction_rank': args.interaction_rank, 'use_codebook_gate': config['use_codebook_gate'], 'initial_gate_probability': args.initial_gate_probability, 'lambda_relation_kd': config['lambda_relation_kd'], 'lambda_boundary': config['lambda_boundary'], 'temperature': args.temperature, 'boundary_margin': args.boundary_margin, 'encoder_learning_rate': args.encoder_learning_rate, 'module_learning_rate': args.module_learning_rate, 'freeze_encoder_epochs': args.freeze_encoder_epochs, 'epochs': args.epochs, 'checkpoint_selection': 'minimum development gold multitask loss', 'architecture_selection': 'development macro-AP with pre-specified gates', 'threshold_selection': 'per-aspect development F1; conservative project tie-break', 'test_data_read': False, 'rating_used_as_target': False, 'pos_weights': pos_weights, 'aspect_balance': aspect_balance.detach().cpu().tolist(), 'primary_boundary_edges': [[aspects[a], aspects[b]] for a, b in primary_edges], 'device': str(device), 'amp': use_amp, 'data_dir': str(args.data_dir)}
    repair.write_json(out_dir / 'train_config.json', run_config)
    repair.write_json(out_dir / 'parameter_counts.json', repair.model_parameter_counts(model))
    optimizer = torch.optim.AdamW(parameter_groups(model, args.encoder_learning_rate, args.module_learning_rate, args.weight_decay))
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(round(total_updates * args.warmup_ratio)), num_training_steps=total_updates)
    scaler = make_scaler(use_amp)
    history: list[dict[str, Any]] = []
    best_loss = float('inf')
    best_epoch = 0
    patience = 0
    training_start = time.perf_counter()
    best_model_dir = out_dir / 'best_model'
    for epoch in range(1, args.epochs + 1):
        encoder_trainable = epoch > args.freeze_encoder_epochs
        set_encoder_trainable(model, encoder_trainable)
        model.train()
        if not encoder_trainable:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        sums = {'train_total_loss': 0.0, 'train_gold_loss': 0.0, 'train_relation_kd_loss': 0.0, 'train_boundary_loss': 0.0}
        encoder_grad_sum = 0.0
        module_grad_sum = 0.0
        optimizer_steps = 0
        n_batches = 0
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(train_loader, start=1):
            inputs, labels = split_batch(batch, device)
            row_indices = labels['row_index'].long()
            with autocast_context(use_amp):
                outputs = model(**inputs)
                gold_loss, _ = repair.gold_multitask_loss(outputs, labels['labels_mention'], labels['labels_positive'], labels['labels_negative'], pos_weights, aspect_balance)
                relation_value = outputs['mention_logit'].sum() * 0.0
                if teacher_train is not None:
                    teacher_batch = teacher_train[row_indices].to(device, non_blocking=True)
                    relation_value, _ = repair.relation_kd_loss(outputs, teacher_batch, args.temperature)
                boundary_value = outputs['mention_logit'].sum() * 0.0
                if primary_edges:
                    boundary_value = repair.boundary_ranking_loss(outputs['mention_logit'], labels['labels_mention'], primary_edges, args.boundary_margin)
                total_loss = gold_loss + config['lambda_relation_kd'] * relation_value + config['lambda_boundary'] * boundary_value
                scaled = total_loss / args.gradient_accumulation_steps
            scaler.scale(scaled).backward()
            should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                encoder_grad_sum += gradient_norm(model.encoder.parameters())
                module_grad_sum += gradient_norm((parameter for name, parameter in model.named_parameters() if not name.startswith('encoder.')))
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
            sums['train_total_loss'] += float(total_loss.detach().cpu())
            sums['train_gold_loss'] += float(gold_loss.detach().cpu())
            sums['train_relation_kd_loss'] += float(relation_value.detach().cpu())
            sums['train_boundary_loss'] += float(boundary_value.detach().cpu())
            n_batches += 1
        dev_losses = compute_dev_gold_loss(model, dev_loader, device, use_amp, pos_weights, aspect_balance)
        improved = dev_losses['gold_total_loss'] < best_loss - args.min_delta
        if improved:
            best_loss = dev_losses['gold_total_loss']
            best_epoch = epoch
            patience = 0
            metadata = {**run_config, 'best_epoch': best_epoch, 'best_dev_gold_multitask_loss': best_loss, 'interaction_rank': args.interaction_rank, 'dropout': float(getattr(model.encoder.config, 'hidden_dropout_prob', 0.1))}
            repair.save_student_bundle(model, tokenizer, best_model_dir, metadata)
        else:
            patience += 1
        query_frame, query_summary = repair.query_diagnostics(model, aspects)
        lr_by_name = {group.get('group_name', f'group_{i}'): float(group['lr']) for i, group in enumerate(optimizer.param_groups)}
        row = {'epoch': epoch, **{key: value / max(n_batches, 1) for key, value in sums.items()}, 'dev_gold_total_loss': dev_losses['gold_total_loss'], 'dev_mention_loss': dev_losses['mention_loss'], 'dev_positive_loss': dev_losses['positive_loss'], 'dev_negative_loss': dev_losses['negative_loss'], 'encoder_trainable': int(encoder_trainable), 'encoder_learning_rate': max([value for name, value in lr_by_name.items() if name.startswith('encoder_')] or [0.0]), 'module_learning_rate': max([value for name, value in lr_by_name.items() if name.startswith('module_')] or [0.0]), 'encoder_grad_norm_mean': encoder_grad_sum / max(optimizer_steps, 1), 'module_grad_norm_mean': module_grad_sum / max(optimizer_steps, 1), 'gate_alpha_mean': query_summary['gate_mean'], 'gate_alpha_min': query_summary['gate_min'], 'gate_alpha_max': query_summary['gate_max'], 'query_cosine_offdiag_mean': query_summary['query_cosine_offdiag_mean'], 'query_cosine_offdiag_max': query_summary['query_cosine_offdiag_max'], 'improved': int(improved), 'patience_count': patience, 'optimizer_steps': optimizer_steps, 'epoch_seconds': time.perf_counter() - epoch_start}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / 'training_history.csv', index=False, encoding='utf-8-sig')
        query_frame.to_csv(out_dir / f'query_diagnostics_epoch_{epoch:02d}.csv', index=False, encoding='utf-8-sig')
        print(pd.DataFrame([row]).to_string(index=False))
        if patience >= args.early_stopping_patience:
            break
    if best_epoch <= 0 or not best_model_dir.exists():
        raise RuntimeError('No best checkpoint was saved')
    training_seconds = time.perf_counter() - training_start
    tokenizer, best_model, _ = repair.load_student_bundle(best_model_dir, device)
    ap_summary, dev_inference_seconds, query_summary = evaluate_dev(best_model, tokenizer, dev_rows, args, device, use_amp, out_dir)
    peak_memory = int(torch.cuda.max_memory_allocated()) if device.type == 'cuda' else 0
    efficiency = pd.DataFrame([{'model_variant': args.variant, 'encoder_source': args.encoder_source, 'run_name': run_name, 'seed': args.seed, **repair.model_parameter_counts(best_model), 'best_epoch': best_epoch, 'training_seconds': training_seconds, 'dev_inference_seconds': dev_inference_seconds, 'cuda_peak_memory_bytes': peak_memory, 'full_encoder_calls_per_review': 1, 'dev_mean_three_tasks_macro_ap': ap_summary['mean_three_tasks'], **query_summary}])
    efficiency.to_csv(out_dir / 'efficiency_metrics.csv', index=False, encoding='utf-8-sig')
    marker.write_text(f'variant={args.variant}\nencoder_source={args.encoder_source}\nseed={args.seed}\nbest_epoch={best_epoch}\ntest_evaluated=0\n', encoding='utf-8')
    print(efficiency.to_string(index=False))
    print(f'[DONE] {out_dir}')
if __name__ == '__main__':
    main()
