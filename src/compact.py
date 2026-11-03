from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
N_ASPECTS = 14

def export_compact(source_dir, output_dir):
    from .student import load_student_bundle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, source, metadata = load_student_bundle(source_dir, device)
    assert not source.use_codebook_gate
    model = CompactDCQStudent(source.encoder, interaction_rank=source.interaction_rank).to(device)
    with torch.no_grad():
        model.learned_queries.copy_(source.learned_queries)
        model.aspect_task_bias.copy_(source.aspect_task_bias)
    for name in ['query_norm', 'query_proj', 'key_proj', 'value_proj', 'interaction', 'task_out']:
        getattr(model, name).load_state_dict(getattr(source, name).state_dict())
    assert sum(p.numel() for p in model.parameters()) == 82721325
    save_compact_bundle(model, tokenizer, output_dir, metadata)

class CompactDCQStudent(nn.Module):

    def __init__(self, encoder: nn.Module, interaction_rank: int=192, dropout: float=0.1, n_aspects: int=N_ASPECTS) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_aspects = int(n_aspects)
        self.interaction_rank = int(interaction_rank)
        hidden_size = int(encoder.config.hidden_size)
        self.learned_queries = nn.Parameter(torch.empty(self.n_aspects, hidden_size))
        nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.query_proj = nn.Linear(hidden_size, self.interaction_rank, bias=False)
        self.key_proj = nn.Linear(hidden_size, self.interaction_rank, bias=False)
        self.value_proj = nn.Linear(hidden_size, self.interaction_rank, bias=False)
        self.interaction = nn.Sequential(nn.Linear(self.interaction_rank * 4, self.interaction_rank), nn.GELU(), nn.Dropout(float(dropout)))
        self.task_out = nn.Linear(self.interaction_rank, 3)
        self.aspect_task_bias = nn.Parameter(torch.zeros(self.n_aspects, 3))

    def query_hidden_states(self) -> torch.Tensor:
        return self.query_norm(self.learned_queries)

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

def model_parameter_counts(model: CompactDCQStudent) -> dict[str, int]:
    total = sum((parameter.numel() for parameter in model.parameters()))
    trainable = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    encoder = sum((parameter.numel() for parameter in model.encoder.parameters()))
    return {'total_parameters': int(total), 'trainable_parameters': int(trainable), 'encoder_parameters': int(encoder), 'dcq_module_parameters': int(total - encoder)}

def save_compact_bundle(model: CompactDCQStudent, tokenizer: Any, output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(output_dir / 'encoder'))
    tokenizer.save_pretrained(str(output_dir))
    modules = {'learned_queries': model.learned_queries.detach().cpu(), 'query_norm': model.query_norm.state_dict(), 'query_proj': model.query_proj.state_dict(), 'key_proj': model.key_proj.state_dict(), 'value_proj': model.value_proj.state_dict(), 'interaction': model.interaction.state_dict(), 'task_out': model.task_out.state_dict(), 'aspect_task_bias': model.aspect_task_bias.detach().cpu()}
    torch.save(modules, output_dir / 'dcq_compact_modules.pt')
    (output_dir / 'model_bundle_config.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

def load_compact_bundle(model_dir: Path, device: torch.device) -> tuple[Any, CompactDCQStudent, dict[str, Any]]:
    metadata = json.loads((model_dir / 'model_bundle_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(model_dir / 'encoder'), local_files_only=True)
    try:
        modules = torch.load(model_dir / 'dcq_compact_modules.pt', map_location='cpu', weights_only=True)
    except TypeError:
        modules = torch.load(model_dir / 'dcq_compact_modules.pt', map_location='cpu')
    model = CompactDCQStudent(encoder=encoder, interaction_rank=int(metadata['interaction_rank']), dropout=float(metadata.get('dropout', 0.1)), n_aspects=int(metadata.get('n_aspect_queries', N_ASPECTS)))
    with torch.no_grad():
        model.learned_queries.copy_(modules['learned_queries'])
        model.aspect_task_bias.copy_(modules['aspect_task_bias'])
    model.query_norm.load_state_dict(modules['query_norm'])
    model.query_proj.load_state_dict(modules['query_proj'])
    model.key_proj.load_state_dict(modules['key_proj'])
    model.value_proj.load_state_dict(modules['value_proj'])
    model.interaction.load_state_dict(modules['interaction'])
    model.task_out.load_state_dict(modules['task_out'])
    model.to(device)
    return (tokenizer, model, metadata)
