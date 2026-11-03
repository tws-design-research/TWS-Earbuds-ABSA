import argparse
from pathlib import Path
import numpy as np
from common import ROOT, DATA, SEEDS, ASPECTS, read_jsonl


def score_reviews(model_dir, texts, batch_size=32, corpus_mode=True):
    import torch
    from src.student import load_student_bundle
    from src.compact import load_compact_bundle
    from src.absa import sigmoid_numpy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loader = load_compact_bundle if (model_dir / 'dcq_compact_modules.pt').exists() else load_student_bundle
    tokenizer, model, _ = loader(model_dir, device)
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[start:start + batch_size], truncation=True, padding=True,
                               max_length=384, return_tensors='pt')
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(**{k: v.to(device) for k, v in inputs.items()})
            if corpus_mode:
                m = torch.sigmoid(logits['mention_logit']).float().cpu().numpy()
                scores = np.stack([m, m * torch.sigmoid(logits['positive_logit']).float().cpu().numpy(),
                                      m * torch.sigmoid(logits['negative_logit']).float().cpu().numpy()], axis=-1)
            else:
                m = sigmoid_numpy(logits['mention_logit'].float().cpu().numpy())
                scores = np.stack([m, m * sigmoid_numpy(logits['positive_logit'].float().cpu().numpy()),
                                      m * sigmoid_numpy(logits['negative_logit'].float().cpu().numpy())], axis=-1)
            output.append(scores)
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=ROOT / 'trained/compact_students')
    parser.add_argument('--output', type=Path, default=ROOT / 'predicted_corpus.npz')
    args = parser.parse_args()
    rows = read_jsonl(DATA / 'corpus_text.jsonl.gz')
    import pandas as pd
    formal_rows = pd.read_csv(DATA / 'corpus.csv.gz', usecols=['row_number']).row_number.to_numpy(int) - 1
    assert [r['row_number'] for r in rows] == list(range(1, 309203))
    predictions = []
    for seed in SEEDS:
        folder = args.models / f'seed_{seed:04d}'
        scores = score_reviews(folder / 'best_model', [r['review_text_for_model'] for r in rows])
        threshold = np.load(folder / 'thresholds.npy')
        pred = scores >= threshold
        pred[..., 1:] &= pred[..., :1]
        predictions.append(pred[formal_rows].astype(np.uint8))
    np.savez_compressed(args.output, predictions=np.stack(predictions), seeds=SEEDS,
        row_number=formal_rows + 1, aspects=ASPECTS,
        tasks=['mention', 'positive_signal', 'negative_signal'])


if __name__ == '__main__':
    main()
