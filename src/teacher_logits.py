from __future__ import annotations
import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
BASE = Path(__file__).resolve().parents[1]
MODELING = BASE / '07-modeling'
SCRIPT_DIR = Path(__file__).resolve().parent

def load_module(filename: str, name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
project = load_module('absa.py', 'matched10_teacher_project')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model_dir', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--pair_dir', type=Path, default=BASE / 'trained' / 'inputs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()

def make_loader(rows, tokenizer, args) -> DataLoader:
    return DataLoader(project.RawPairDataset(project.rows_with_indices(rows)), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=project.PairBatchCollator(tokenizer, args.max_length), pin_memory=torch.cuda.is_available())

def split_batch(batch: dict[str, Any], device: torch.device):
    row_index = batch.pop('row_index').cpu().numpy().astype(int)
    for key in ['labels_mention', 'labels_positive', 'labels_negative', 'aspect_index']:
        batch.pop(key, None)
    inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return (inputs, row_index)

def export_split(split: str, rows, tokenizer, model, device, args) -> dict[str, Any]:
    loader = make_loader(rows, tokenizer, args)
    use_amp = device.type == 'cuda' and (not args.no_amp)
    logits = np.empty((len(rows), 3), dtype=np.float32)
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs, indices = split_batch(batch, device)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(**inputs)
            else:
                out = model(**inputs)
            logits[indices, 0] = out['mention_logit'].float().cpu().numpy()
            logits[indices, 1] = out['positive_logit'].float().cpu().numpy()
            logits[indices, 2] = out['negative_logit'].float().cpu().numpy()
    seconds = time.perf_counter() - start
    by_review: dict[str, np.ndarray] = {}
    aspect_names: dict[int, str] = {}
    for i, row in enumerate(rows):
        annotation_id = str(row['annotation_id'])
        aspect_index = int(row['aspect_index'])
        aspect_names[aspect_index] = str(row['aspect'])
        matrix = by_review.setdefault(annotation_id, np.full((14, 3), np.nan, dtype=np.float32))
        if not np.isnan(matrix[aspect_index]).all():
            raise RuntimeError(f'Duplicate teacher cell: {annotation_id}/{aspect_index}')
        matrix[aspect_index] = logits[i]
    annotation_ids = list(by_review)
    review_logits = np.stack([by_review[x] for x in annotation_ids])
    if review_logits.shape[1:] != (14, 3) or np.isnan(review_logits).any():
        raise RuntimeError(f'Incomplete {split} teacher matrix: {review_logits.shape}')
    aspects = [aspect_names[i] for i in range(14)]
    out_path = args.out_dir / f'teacher_{split}_logits.npz'
    np.savez_compressed(out_path, annotation_ids=np.asarray(annotation_ids, dtype=str), aspects=np.asarray(aspects, dtype=str), logits=review_logits)
    return {'seed': args.seed, 'split': split, 'pair_rows': len(rows), 'review_rows': len(annotation_ids), 'shape': 'x'.join(map(str, review_logits.shape)), 'seconds': seconds, 'output': str(out_path), 'status': 'PASS'}

def main() -> None:
    args = parse_args()
    marker = args.out_dir / 'MATCHED10_TRAIN_DEV_TEACHER_LOGITS_COMPLETE.txt'
    if marker.exists() and (not args.force):
        print(f'[SKIP] Existing matched teacher logits: {args.out_dir}')
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, model, metadata = project.load_model_bundle(args.model_dir, device)
    if int(metadata.get('seed', args.seed)) != args.seed:
        raise RuntimeError(f"Teacher metadata seed mismatch: metadata={metadata.get('seed')}, arg={args.seed}")
    records = []
    for split in ('train', 'dev'):
        rows = project.read_jsonl(args.pair_dir / f'{split}_aspect_pairs.jsonl')
        records.append(export_split(split, rows, tokenizer, model, device, args))
    pd.DataFrame(records).to_csv(args.out_dir / 'matched10_teacher_export_audit.csv', index=False, encoding='utf-8-sig')
    config = {'teacher': 'M2_shared_aspect_balanced', 'seed': args.seed, 'model_dir': str(args.model_dir), 'model_metadata': metadata, 'splits_exported': ['train', 'dev'], 'test_exported': False, 'device': str(device)}
    (args.out_dir / 'matched10_teacher_export_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    marker.write_text(f'seed={args.seed}\ntest_exported=0\n', encoding='utf-8')
    print(pd.DataFrame(records).to_string(index=False))
    print(f'[DONE] seed={args.seed}: {args.out_dir}')
if __name__ == '__main__':
    main()
