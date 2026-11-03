import logging
from tqdm import tqdm
from . import tokenization
logger = logging.getLogger(__name__)

class InputFeatures:

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids, self.input_mask = (input_ids, input_mask)
        self.segment_ids, self.label_id = (segment_ids, label_id)

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    features = []
    for ex_index, example in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        elif len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:max_seq_length - 2]
        tokens = []
        segment_ids = []
        tokens.append('[CLS]')
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        label_id = label_map[example.label]
        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
