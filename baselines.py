import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from common import ROOT, SEEDS, TASKS, ASPECTS, labelled, pair_rows, tune, metrics


def tfidf(output):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    rows = {s: labelled(s) for s in ['train', 'dev', 'test']}
    labels = {s: np.array([[r[k] for k in ['y_mention', 'y_positive_signal', 'y_negative_signal']] for r in v]).transpose(0, 2, 1) for s, v in rows.items()}
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                max_features=80000, sublinear_tf=True)
    train = vectorizer.fit_transform([r['review_text_for_model'] for r in rows['train']])
    features = {s: vectorizer.transform([r['review_text_for_model'] for r in rows[s]]) for s in ['dev', 'test']}
    probabilities = {s: np.zeros((len(rows[s]), 14, 3)) for s in ['dev', 'test']}
    for a in range(14):
        for k in range(3):
            mask = np.ones(len(rows['train']), bool) if k == 0 else labels['train'][:, a, 0].astype(bool)
            model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=20)
            model.fit(train[mask], labels['train'][mask, a, k])
            for s in features:
                probabilities[s][:, a, k] = model.predict_proba(features[s])[:, 1]
    for s in probabilities:
        probabilities[s][..., 1:] *= probabilities[s][..., :1]
    threshold = tune(labels['dev'], probabilities['dev'])
    result = pd.DataFrame(metrics(labels['test'], probabilities['test'] >= threshold, probabilities['test'], 'tfidf_lr', 20))
    result.to_csv(output / 'tfidf_performance.csv', index=False)
    np.savez_compressed(output / 'tfidf_scores.npz', dev=probabilities['dev'], test=probabilities['test'], thresholds=threshold)
    actual = [result[result.task.eq(t)].f1.mean() for t in TASKS]
    np.testing.assert_array_equal(np.round(actual, 4), [.7638, .5958, .5060])
    assert round(result[result.task.eq('negative') & result.aspect.isin(ASPECTS[:10])].f1.mean(), 4) == .5274
    print('TF-IDF + LR matches the manuscript.', flush=True)


def aspect_name(output, base_model, seeds):
    from src import aspect_name as engine
    engine.HF_MODEL_DIR = base_model
    engine.INTERMEDIATE_ROOT = output / 'aspect_name'
    engine.TRAINING_WORK_ROOT = output / 'aspect_name_training'
    pairs = {s: pair_rows(s, names_only=True) for s in ['train', 'dev', 'test']}
    for seed in seeds:
        for task in ['y_mention', 'y_positive_signal', 'y_negative_signal']:
            engine.train_transformer_task(seed, task, pairs['train'], pairs['dev'], pairs['test'])


def qa_b(output, backbone, seeds):
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, RandomSampler, TensorDataset
    from src.qab.modeling import BertConfig, BertForSequenceClassification
    from src.qab.optimization import BERTAdam
    from src.qab.adapter import dataset
    datasets = {split: dataset(pair_rows(split), backbone) for split in ['train', 'dev', 'test']}
    assert len(datasets['train']) == 206150
    source = load_file(str(backbone / 'model.safetensors'), device='cpu')
    converted = {k[5:].replace('.LayerNorm.weight', '.LayerNorm.gamma').replace('.LayerNorm.bias', '.LayerNorm.beta'): v
                 for k, v in source.items() if k.startswith('bert.')}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = BertForSequenceClassification(BertConfig.from_json_file(str(backbone / 'config.json')), 2)
        model.bert.load_state_dict(converted, strict=True)
        model.to(device)
        no_decay = ['bias', 'gamma', 'beta']
        groups = [{'params': [p for n, p in model.named_parameters() if not any(s in n for s in no_decay)], 'weight_decay_rate': .01},
                  {'params': [p for n, p in model.named_parameters() if any(s in n for s in no_decay)], 'weight_decay_rate': 0.}]
        optimizer = BERTAdam(groups, lr=2e-5, warmup=.1, t_total=int(206150 / 24 * 4))
        loader = DataLoader(datasets['train'], sampler=RandomSampler(datasets['train']), batch_size=24)
        for epoch in range(4):
            model.train()
            for batch in loader:
                ids, mask, segments, labels = [x.to(device) for x in batch]
                loss, _ = model(ids, segments, mask, labels)
                loss.backward(); optimizer.step(); model.zero_grad()
            print(f'QA-B seed {seed}: epoch {epoch + 1}/4', flush=True)
        folder = output / f'qa_b/seed_{seed:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), folder / 'epoch4.pt')
        model.eval()
        for split in ['dev', 'test']:
            scores = []
            with torch.inference_mode():
                for batch in DataLoader(datasets[split], batch_size=32):
                    ids, mask, segments, _ = [x.to(device) for x in batch]
                    logits = model(ids, segments, mask)
                    scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            np.savez_compressed(folder / f'{split}_yes_scores.npz', yes=np.concatenate(scores).reshape(-1, 5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['tfidf', 'aspect-name', 'qa-b'], default='tfidf')
    parser.add_argument('--base-model', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.model == 'tfidf':
        tfidf(args.output)
    elif args.base_model is None:
        parser.error('--base-model is required for neural baselines')
    elif args.model == 'aspect-name':
        aspect_name(args.output, args.base_model, args.seeds)
    else:
        qa_b(args.output, args.base_model, args.seeds)
