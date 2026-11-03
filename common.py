from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SEEDS = [20, 3, 15, 12, 13, 21, 30, 17, 6, 22]
ASPECTS = [
    'sound_quality', 'noise_control', 'fit_comfort', 'battery_charging',
    'connectivity_pairing', 'controls_interaction', 'call_mic',
    'durability_reliability', 'charging_case_design', 'appearance_materials',
    'waterproof_sport', 'app_ecosystem', 'price_value',
    'service_shipping_packaging',
]
TASKS = ['mention', 'positive', 'negative']


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def labelled(split):
    return read_jsonl(DATA / f'{split}.jsonl.gz')


def pair_rows(split, names_only=False, aspect_definitions=None):
    if not names_only and aspect_definitions is None:
        raise ValueError('Aspect definitions are required for name-definition prompts.')
    result = []
    for i, review in enumerate(labelled(split), 2):
        for a, name in enumerate(ASPECTS):
            m, p, n = [int(review[k][a]) for k in ['y_mention', 'y_positive_signal', 'y_negative_signal']]
            raw = 'not_mentioned' if not m else 'mixed' if p and n else 'pos' if p else 'neg' if n else 'neutral'
            result.append(dict(annotation_id=review['annotation_id'], parent_asin=review['parent_asin'],
                asin=review['asin'], rating=review['rating'], split=split, csv_row_number=i,
                aspect_index=a, aspect=name, aspect_level=review['aspect_level'][a],
                include_in_core_design_priority=str(review['include_in_core_design_priority'][a]),
                review_text_for_model=review['review_text_for_model'],
                aspect_prompt=f'Aspect: {name}.' if names_only else f'Aspect: {name}. Definition: {aspect_definitions[name]}',
                raw_label=raw, y_mention=m, y_positive_signal=p, y_negative_signal=n,
                y_neutral=int(review['y_neutral'][a]),
                prompt_variant='aspect_name_only' if names_only else 'name_definition'))
    return result


def prepare_inputs(destination, aspect_definitions=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'dev', 'test']:
        for suffix, rows in [('aspect_pairs', pair_rows(split, aspect_definitions=aspect_definitions)), ('reviews_14aspect', labelled(split))]:
            path = destination / f'{split}_{suffix}.jsonl'
            with path.open('w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return destination


def tune(y, scores):
    thresholds = np.empty((14, 3))
    grid = np.round(np.arange(.01, 1., .01), 2)
    for a in range(14):
        for k in range(3):
            pred = scores[:, a, k, None] >= grid
            gold = y[:, a, k, None].astype(bool)
            tp = (pred & gold).sum(axis=0)
            fp = (pred & ~gold).sum(axis=0)
            fn = (~pred & gold).sum(axis=0)
            def ratio(n, d):
                return np.divide(n, d, out=np.zeros(len(grid)), where=d != 0)
            precision = ratio(tp, tp + fp)
            recall = ratio(tp, tp + fn)
            f1 = ratio(2 * tp, 2 * tp + fp + fn)
            j = max(range(len(grid)), key=lambda i: (f1[i], precision[i], recall[i], grid[i]))
            thresholds[a, k] = grid[j]
    return thresholds


def metrics(y, pred, score, model, seed):
    rows = []
    for a, name in enumerate(ASPECTS):
        for k, task in enumerate(TASKS):
            rows.append(dict(model=model, seed=seed, aspect=name, task=task,
                support=int(y[:, a, k].sum()),
                f1=f1_score(y[:, a, k], pred[:, a, k], zero_division=0),
                ap=average_precision_score(y[:, a, k], score[:, a, k])))
    return rows


def corpus():
    frame = pd.read_csv(DATA / 'corpus.csv.gz', keep_default_na=False)
    archive = np.load(DATA / 'corpus_predictions.npz')
    assert np.array_equal(archive['seeds'], SEEDS)
    assert np.array_equal(archive['row_number'], frame.row_number)
    pred = archive['predictions']
    assert pred.shape == (10, 305350, 14, 3)
    assert np.isin(pred, [0, 1]).all()
    assert (pred[..., 1:] <= pred[..., :1]).all()
    votes = pred.mean(axis=0)
    for a, name in enumerate(ASPECTS):
        for k, task in enumerate(TASKS):
            frame[f'{name}__{task}_vote_rate'] = votes[:, a, k]
    return frame, pred, votes
