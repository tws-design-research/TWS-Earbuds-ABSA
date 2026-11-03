import numpy as np
import torch
from types import SimpleNamespace
from torch.utils.data import TensorDataset
from .features import convert_examples_to_features
from .tokenization import FullTokenizer

STATES = ['positive', 'neutral', 'negative', 'conflict', 'none']
MAPPING = {'pos': 'positive', 'neutral': 'neutral', 'neg': 'negative', 'mixed': 'conflict', 'not_mentioned': 'none'}


def dataset(rows, backbone):
    tokenizer = FullTokenizer(str(backbone / 'vocab.txt'), do_lower_case=True)
    cache = {text: tokenizer.tokenize(text) for text in dict.fromkeys(r['review_text_for_model'] for r in rows)}
    def sentence(aspect, state):
        return f'the polarity of the aspect {aspect.replace("_", " ")} is {state} .'
    for aspect in dict.fromkeys(r['aspect'] for r in rows):
        for state in STATES:
            text = sentence(aspect, state)
            cache[text] = tokenizer.tokenize(text)
    class CachedTokenizer:
        def tokenize(self, text):
            return list(cache[text])
        def convert_tokens_to_ids(self, tokens):
            return tokenizer.convert_tokens_to_ids(tokens)
    examples = [SimpleNamespace(guid=f'{r["annotation_id"]}-{r["aspect"]}-{state}',
        text_a=sentence(r['aspect'], state), text_b=r['review_text_for_model'],
        label=str(int(state == MAPPING[r['raw_label']]))) for r in rows for state in STATES]
    features = convert_examples_to_features(examples, ['0', '1'], 128, CachedTokenizer())
    return TensorDataset(*[torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        for k in ['input_ids', 'input_mask', 'segment_ids', 'label_id']])


def decode(yes):
    yes = np.asarray(yes, dtype=float)
    bits = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 0, 0]])
    q = yes / yes.sum(axis=-1, keepdims=True)
    return bits[yes.argmax(axis=-1)], np.stack([1 - q[..., 4], q[..., 0] + q[..., 3], q[..., 2] + q[..., 3]], axis=-1)
