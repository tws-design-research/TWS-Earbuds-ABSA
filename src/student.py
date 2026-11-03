from __future__ import annotations
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
N_ASPECTS = 14

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

class ReviewDataset(Dataset):

    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        row['__row_index__'] = index
        return row

class ReviewBatchCollator:

    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer([str(row['review_text_for_model']) for row in batch], truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        encoded['labels_mention'] = torch.tensor([row['y_mention'] for row in batch], dtype=torch.float32)
        encoded['labels_positive'] = torch.tensor([row['y_positive_signal'] for row in batch], dtype=torch.float32)
        encoded['labels_negative'] = torch.tensor([row['y_negative_signal'] for row in batch], dtype=torch.float32)
        encoded['row_index'] = torch.tensor([int(row['__row_index__']) for row in batch], dtype=torch.long)
        return encoded

class DCQRepairStudent(nn.Module):

    def __init__(self, encoder: nn.Module, codebook_anchors: torch.Tensor, interaction_rank: int=192, dropout: float=0.1, use_codebook_gate: bool=False, initial_gate_probability: float=0.01) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        if codebook_anchors.shape != (N_ASPECTS, hidden_size):
            raise ValueError(f'Expected anchors {(N_ASPECTS, hidden_size)}, got {tuple(codebook_anchors.shape)}')
        self.interaction_rank = int(interaction_rank)
        self.use_codebook_gate = bool(use_codebook_gate)
        self.initial_gate_probability = float(initial_gate_probability)
        self.register_buffer('codebook_anchors', codebook_anchors.detach().clone(), persistent=True)
        self.learned_queries = nn.Parameter(torch.empty(N_ASPECTS, hidden_size))
        nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)
        self.anchor_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.anchor_projection.weight)
        gate_p = min(max(float(initial_gate_probability), 0.0001), 1.0 - 0.0001)
        gate_logit = math.log(gate_p / (1.0 - gate_p))
        self.anchor_gate_logits = nn.Parameter(torch.full((N_ASPECTS, 1), gate_logit, dtype=torch.float32))
        self.query_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, self.interaction_rank, bias=False)
        self.key_proj = nn.Linear(hidden_size, self.interaction_rank, bias=False)
        self.value_proj = nn.Linear(hidden_size, self.interaction_rank, bias=False)
        self.interaction = nn.Sequential(nn.Linear(self.interaction_rank * 4, self.interaction_rank), nn.GELU(), nn.Dropout(float(dropout)))
        self.task_out = nn.Linear(self.interaction_rank, 3)
        self.aspect_task_bias = nn.Parameter(torch.zeros(N_ASPECTS, 3))

    def gate_values(self) -> torch.Tensor:
        if not self.use_codebook_gate:
            return torch.zeros_like(self.anchor_gate_logits)
        return torch.sigmoid(self.anchor_gate_logits)

    def query_hidden_states(self) -> torch.Tensor:
        projected_anchor = self.anchor_projection(self.codebook_anchors)
        query = self.learned_queries + self.gate_values() * projected_anchor
        return self.query_norm(query)

    def forward(self, **encoder_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(**encoder_inputs)
        token_states = outputs.last_hidden_state
        batch_size = token_states.size(0)
        query_hidden = self.query_hidden_states()
        query = self.query_proj(query_hidden)
        key = self.key_proj(token_states)
        value = self.value_proj(token_states)
        attention_logits = torch.einsum('ar,blr->bal', query, key) / math.sqrt(self.interaction_rank)
        attention_mask = encoder_inputs.get('attention_mask')
        if attention_mask is not None:
            valid = attention_mask.bool().unsqueeze(1)
            attention_logits = attention_logits.masked_fill(~valid, torch.finfo(attention_logits.dtype).min)
        attention = torch.softmax(attention_logits, dim=-1)
        context = torch.einsum('bal,blr->bar', attention, value)
        query_batch = query.unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat([context, query_batch, context * query_batch, torch.abs(context - query_batch)], dim=-1)
        hidden = self.interaction(features)
        logits = self.task_out(hidden) + self.aspect_task_bias.unsqueeze(0)
        return {'mention_logit': logits[:, :, 0], 'positive_logit': logits[:, :, 1], 'negative_logit': logits[:, :, 2], 'attention': attention}

def encode_codebook_anchors(encoder: nn.Module, tokenizer: Any, codebook: pd.DataFrame, device: torch.device, max_length: int=128) -> torch.Tensor:
    required = {'aspect', 'definition'}
    missing = required - set(codebook.columns)
    if missing:
        raise ValueError(f'Codebook missing columns: {sorted(missing)}')
    if 'aspect_index' in codebook.columns:
        codebook = codebook.sort_values('aspect_index')
    if len(codebook) != N_ASPECTS:
        raise ValueError(f'Expected {N_ASPECTS} codebook rows, found {len(codebook)}')
    prompts = [f'Aspect: {row.aspect}. Definition: {row.definition}' for row in codebook.itertuples(index=False)]
    encoded = tokenizer(prompts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    encoded = {key: value.to(device) for key, value in encoded.items()}
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        anchors = encoder(**encoded).last_hidden_state[:, 0, :].detach().float().cpu()
    if was_training:
        encoder.train()
    return anchors

def instantiate_repair_student(tokenizer_dir: Path, encoder_dir: Path, codebook: pd.DataFrame, device: torch.device, interaction_rank: int, use_codebook_gate: bool, initial_gate_probability: float) -> tuple[Any, DCQRepairStudent]:
    if not tokenizer_dir.exists():
        raise FileNotFoundError(tokenizer_dir)
    if not encoder_dir.exists():
        raise FileNotFoundError(encoder_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True)
    encoder.to(device)
    if use_codebook_gate:
        anchors = encode_codebook_anchors(encoder, tokenizer, codebook, device)
    else:
        anchors = torch.zeros(N_ASPECTS, int(encoder.config.hidden_size))
    model = DCQRepairStudent(encoder=encoder, codebook_anchors=anchors, interaction_rank=interaction_rank, dropout=float(getattr(encoder.config, 'hidden_dropout_prob', 0.1)), use_codebook_gate=use_codebook_gate, initial_gate_probability=initial_gate_probability)
    model.to(device)
    return (tokenizer, model)

def save_student_bundle(model: DCQRepairStudent, tokenizer: Any, output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(output_dir / 'encoder'))
    tokenizer.save_pretrained(str(output_dir))
    modules = {'codebook_anchors': model.codebook_anchors.detach().cpu(), 'learned_queries': model.learned_queries.detach().cpu(), 'anchor_projection': model.anchor_projection.state_dict(), 'anchor_gate_logits': model.anchor_gate_logits.detach().cpu(), 'query_norm': model.query_norm.state_dict(), 'query_proj': model.query_proj.state_dict(), 'key_proj': model.key_proj.state_dict(), 'value_proj': model.value_proj.state_dict(), 'interaction': model.interaction.state_dict(), 'task_out': model.task_out.state_dict(), 'aspect_task_bias': model.aspect_task_bias.detach().cpu()}
    torch.save(modules, output_dir / 'dcq_repair_modules.pt')
    write_json(output_dir / 'model_bundle_config.json', metadata)

def load_student_bundle(model_dir: Path, device: torch.device) -> tuple[Any, DCQRepairStudent, dict[str, Any]]:
    metadata = json.loads((model_dir / 'model_bundle_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(model_dir / 'encoder'), local_files_only=True)
    try:
        modules = torch.load(model_dir / 'dcq_repair_modules.pt', map_location='cpu', weights_only=True)
    except TypeError:
        modules = torch.load(model_dir / 'dcq_repair_modules.pt', map_location='cpu')
    model = DCQRepairStudent(encoder=encoder, codebook_anchors=modules['codebook_anchors'], interaction_rank=int(metadata['interaction_rank']), dropout=float(metadata.get('dropout', 0.1)), use_codebook_gate=bool(metadata.get('use_codebook_gate', False)), initial_gate_probability=float(metadata.get('initial_gate_probability', 0.01)))
    with torch.no_grad():
        model.learned_queries.copy_(modules['learned_queries'])
        model.anchor_gate_logits.copy_(modules['anchor_gate_logits'])
        model.aspect_task_bias.copy_(modules['aspect_task_bias'])
    model.anchor_projection.load_state_dict(modules['anchor_projection'])
    model.query_norm.load_state_dict(modules['query_norm'])
    model.query_proj.load_state_dict(modules['query_proj'])
    model.key_proj.load_state_dict(modules['key_proj'])
    model.value_proj.load_state_dict(modules['value_proj'])
    model.interaction.load_state_dict(modules['interaction'])
    model.task_out.load_state_dict(modules['task_out'])
    model.to(device)
    return (tokenizer, model, metadata)

def calculate_pos_weights(review_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    mention = np.asarray([row['y_mention'] for row in review_rows], dtype=float)
    positive = np.asarray([row['y_positive_signal'] for row in review_rows], dtype=float)
    negative = np.asarray([row['y_negative_signal'] for row in review_rows], dtype=float)
    active = mention == 1

    def ratio(labels: np.ndarray, mask: np.ndarray | None=None) -> float:
        values = labels if mask is None else labels[mask]
        pos = float(values.sum())
        neg = float(values.size - pos)
        return neg / max(pos, 1.0)
    return {'y_mention': ratio(mention), 'y_positive_signal': ratio(positive, active), 'y_negative_signal': ratio(negative, active)}

def calculate_aspect_balance(review_rows: Sequence[dict[str, Any]]) -> torch.Tensor:
    mention = np.asarray([row['y_mention'] for row in review_rows], dtype=float)
    counts = mention.sum(axis=0)
    if np.any(counts <= 0):
        raise ValueError(f'All aspects need mentioned examples, counts={counts.tolist()}')
    total = float(counts.sum())
    return torch.tensor(total / (N_ASPECTS * counts), dtype=torch.float32)

def weighted_bce(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float) -> torch.Tensor:
    weight = torch.tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction='none')

def gold_multitask_loss(outputs: dict[str, torch.Tensor], labels_mention: torch.Tensor, labels_positive: torch.Tensor, labels_negative: torch.Tensor, pos_weights: dict[str, float], aspect_balance: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mention_loss = weighted_bce(outputs['mention_logit'], labels_mention, pos_weights['y_mention']).mean()
    active = labels_mention.bool()
    balance = aspect_balance.to(outputs['mention_logit'].device).view(1, -1)

    def conditional(logit: torch.Tensor, label: torch.Tensor, task: str) -> torch.Tensor:
        per_cell = weighted_bce(logit, label, pos_weights[task]) * balance
        if not bool(active.any()):
            return logit.sum() * 0.0
        return per_cell[active].mean()
    positive_loss = conditional(outputs['positive_logit'], labels_positive, 'y_positive_signal')
    negative_loss = conditional(outputs['negative_logit'], labels_negative, 'y_negative_signal')
    total = mention_loss + positive_loss + negative_loss
    return (total, {'gold_total_loss': total.detach(), 'mention_loss': mention_loss.detach(), 'positive_loss': positive_loss.detach(), 'negative_loss': negative_loss.detach()})

def relation_kd_loss(outputs: dict[str, torch.Tensor], teacher_logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if teacher_logits.ndim != 3 or teacher_logits.shape[1:] != (N_ASPECTS, 3):
        raise ValueError(f'Teacher logits must be [B,14,3], got {teacher_logits.shape}')
    t = float(temperature)

    def pipeline_log_score(m: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return nn.functional.logsigmoid(m) + nn.functional.logsigmoid(c)
    student_scores = [outputs['mention_logit'], pipeline_log_score(outputs['mention_logit'], outputs['positive_logit']), pipeline_log_score(outputs['mention_logit'], outputs['negative_logit'])]
    teacher_scores = [teacher_logits[:, :, 0], pipeline_log_score(teacher_logits[:, :, 0], teacher_logits[:, :, 1]), pipeline_log_score(teacher_logits[:, :, 0], teacher_logits[:, :, 2])]
    pieces: list[torch.Tensor] = []
    for student, teacher in zip(student_scores, teacher_scores):
        teacher_prob = torch.softmax(teacher / t, dim=-1)
        student_log_prob = torch.log_softmax(student / t, dim=-1)
        pieces.append(nn.functional.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (t * t))
    total = sum(pieces) / len(pieces)
    return (total, {'relation_kd_mention': pieces[0].detach(), 'relation_kd_positive': pieces[1].detach(), 'relation_kd_negative': pieces[2].detach()})

def boundary_ranking_loss(mention_logits: torch.Tensor, labels_mention: torch.Tensor, primary_edges: Sequence[tuple[int, int]], margin: float) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for left, right in primary_edges:
        left_y = labels_mention[:, left]
        right_y = labels_mention[:, right]
        left_only = (left_y == 1) & (right_y == 0)
        right_only = (right_y == 1) & (left_y == 0)
        if bool(left_only.any()):
            pieces.append(nn.functional.relu(float(margin) - mention_logits[left_only, left] + mention_logits[left_only, right]))
        if bool(right_only.any()):
            pieces.append(nn.functional.relu(float(margin) - mention_logits[right_only, right] + mention_logits[right_only, left]))
    if not pieces:
        return mention_logits.sum() * 0.0
    return torch.cat(pieces).mean()

def pipeline_scores_from_logits(mention_logit: np.ndarray, positive_logit: np.ndarray, negative_logit: np.ndarray) -> dict[str, np.ndarray]:
    mention = sigmoid_numpy(mention_logit)
    positive_conditional = sigmoid_numpy(positive_logit)
    negative_conditional = sigmoid_numpy(negative_logit)
    return {'mention_score': mention, 'positive_conditional_score': positive_conditional, 'negative_conditional_score': negative_conditional, 'positive_pipeline_score': mention * positive_conditional, 'negative_pipeline_score': mention * negative_conditional}

def flatten_review_rows(review_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    for review in review_rows:
        for index, aspect in enumerate(review['aspects']):
            pair_rows.append({'annotation_id': review.get('annotation_id', ''), 'parent_asin': review.get('parent_asin', ''), 'asin': review.get('asin', ''), 'split': review.get('split', ''), 'rating': review.get('rating', ''), 'review_text_for_model': review.get('review_text_for_model', ''), 'aspect': aspect, 'aspect_index': index, 'aspect_level': review['aspect_level'][index], 'include_in_core_design_priority': review['include_in_core_design_priority'][index], 'y_mention': int(review['y_mention'][index]), 'y_positive_signal': int(review['y_positive_signal'][index]), 'y_negative_signal': int(review['y_negative_signal'][index]), 'y_neutral': int(review['y_neutral'][index]), 'aspect_prompt': ''})
    return pair_rows

def flatten_scores(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).reshape(-1) for key, value in scores.items()}

def model_parameter_counts(model: DCQRepairStudent) -> dict[str, int]:
    total = sum((parameter.numel() for parameter in model.parameters()))
    trainable = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    encoder = sum((parameter.numel() for parameter in model.encoder.parameters()))
    return {'total_parameters': int(total), 'trainable_parameters': int(trainable), 'encoder_parameters': int(encoder), 'dcq_module_parameters': int(total - encoder)}

def query_diagnostics(model: DCQRepairStudent, aspects: Sequence[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    with torch.no_grad():
        queries = model.query_hidden_states().detach().float().cpu()
        gates = model.gate_values().detach().float().cpu().view(-1)
        anchors = model.codebook_anchors.detach().float().cpu()
        qn = nn.functional.normalize(queries, dim=-1)
        cosine = qn @ qn.T
        an = nn.functional.normalize(anchors + 1e-12, dim=-1)
        anchor_cosine = an @ an.T
    rows: list[dict[str, Any]] = []
    for i, aspect in enumerate(aspects):
        q_values = cosine[i].clone()
        q_values[i] = -2.0
        nearest = int(torch.argmax(q_values).item())
        if model.use_codebook_gate:
            a_values = anchor_cosine[i].clone()
            a_values[i] = -2.0
            anchor_nearest = int(torch.argmax(a_values).item())
            nearest_anchor_aspect = aspects[anchor_nearest]
            max_anchor_cosine = float(a_values[anchor_nearest])
        else:
            nearest_anchor_aspect = ''
            max_anchor_cosine = float('nan')
        rows.append({'aspect': aspect, 'aspect_index': i, 'gate_alpha': float(gates[i]), 'query_norm': float(queries[i].norm()), 'nearest_query_aspect': aspects[nearest], 'max_query_cosine': float(q_values[nearest]), 'nearest_anchor_aspect': nearest_anchor_aspect, 'max_anchor_cosine': max_anchor_cosine})
    off_diag = cosine[~torch.eye(N_ASPECTS, dtype=torch.bool)]
    summary = {'gate_mean': float(gates.mean()), 'gate_min': float(gates.min()), 'gate_max': float(gates.max()), 'query_cosine_offdiag_mean': float(off_diag.mean()), 'query_cosine_offdiag_std': float(off_diag.std(unbiased=False)), 'query_cosine_offdiag_max': float(off_diag.max())}
    return (pd.DataFrame(rows), summary)
