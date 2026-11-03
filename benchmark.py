import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from common import ROOT, labelled, pair_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--student', type=Path)
    parser.add_argument('--qa-checkpoint', type=Path)
    parser.add_argument('--bert-base', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'timing.csv')
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from src.absa import load_model_bundle
    from src.compact import load_compact_bundle
    from src.qab.modeling import BertConfig, BertForSequenceClassification
    from src.qab.adapter import dataset, decode
    pairs = pair_rows('test')
    reviews = labelled('test')
    records = []
    for name, path in [('teacher', args.teacher), ('student', args.student), ('qa_b', args.qa_checkpoint)]:
        if path is None:
            continue
        if name == 'qa_b':
            assert args.bert_base is not None
            model = BertForSequenceClassification(BertConfig.from_json_file(str(args.bert_base / 'config.json')), 2)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            model.to('cuda')
            def operation(rows):
                data = dataset(rows, args.bert_base)
                scores = []
                with torch.no_grad():
                    for ids, mask, seg, _ in DataLoader(data, batch_size=32, shuffle=False):
                        scores.append(model(ids.to('cuda'), seg.to('cuda'), mask.to('cuda')).softmax(-1)[:, 1].cpu().numpy())
                yes = np.concatenate(scores).reshape(-1, 5)
                decode(yes)
                return yes
            def run():
                return operation(pairs)
            warm_data = dataset(pairs[:14], args.bert_base)
            def warm():
                ids, mask, seg, _ = next(iter(DataLoader(warm_data, batch_size=32)))
                with torch.no_grad():
                    model(ids.to('cuda'), seg.to('cuda'), mask.to('cuda'))
        else:
            tokenizer, model, _ = (load_model_bundle if name == 'teacher' else load_compact_bundle)(path, torch.device('cuda'))
            inputs = pairs if name == 'teacher' else reviews
            def batch_forward(batch):
                texts = [r['review_text_for_model'] for r in batch]
                prompts = [r['aspect_prompt'] for r in batch] if name == 'teacher' else None
                encoded = tokenizer(texts, prompts, truncation=True, padding=True, max_length=384, return_tensors='pt')
                with torch.no_grad():
                    out = model(**{k: v.to('cuda') for k, v in encoded.items()})
                    return torch.stack([out[t + '_logit'].sigmoid() for t in ['mention', 'positive', 'negative']], dim=-1).reshape(-1, 3).cpu().numpy()
            def run():
                scores = np.concatenate([batch_forward(inputs[i:i + 32]) for i in range(0, len(inputs), 32)])
                scores[:, 1:] *= scores[:, :1]
                return scores
            def warm():
                batch_forward(inputs[:32])
        model.eval()
        for _ in range(2):
            warm(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(1, 6):
            torch.cuda.synchronize()
            start = time.perf_counter()
            run()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            records.append(dict(model=name, repeat=repeat, seconds=seconds, reviews_per_second=583 / seconds,
                parameters=sum(p.numel() for p in model.parameters()),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name()))
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
