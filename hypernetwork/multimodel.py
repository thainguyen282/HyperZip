"""A shared document- and model-conditioned LoRA generator."""
import hashlib
import json
import random

import torch
from torch import nn
from torch.nn import functional as F


class MultiModelHypernetwork(nn.Module):
    """Shared trunk with a learned model identity and shape-specific output heads."""
    def __init__(self, specs, native_dim=1536, context_dim=512, identity_dim=32,
                 width=64, depth=2, rank=4, alpha=4):
        super().__init__()
        if set(specs) != {'3b', '8b'}:
            raise ValueError('Expected exactly 3b and 8b target specifications')
        if min(native_dim, context_dim, identity_dim, width, depth, rank) < 1 or alpha <= 0:
            raise ValueError('Dimensions, rank, and alpha must be positive')
        self.specs = specs
        self.rank, self.alpha = rank, alpha
        self.model_ids = {'3b': 0, '8b': 1}
        self.module_names = sorted({s['module'] for spec in specs.values() for s in spec})
        if not self.module_names or any(not spec for spec in specs.values()):
            raise ValueError('Both models require target projections')
        self.project = nn.Linear(native_dim, context_dim)
        self.model_embedding = nn.Embedding(2, identity_dim)
        self.layer_embedding = nn.Embedding(1 + max(s['layer'] for spec in specs.values() for s in spec), identity_dim)
        self.module_embedding = nn.Embedding(len(self.module_names), identity_dim)
        dimensions = [context_dim + 3 * identity_dim] + [width] * depth
        self.trunk = nn.Sequential(*(item for a, b in zip(dimensions, dimensions[1:])
                                     for item in (nn.Linear(a, b), nn.SiLU())))
        self.heads = nn.ModuleDict()
        for spec in specs.values():
            for s in spec:
                key = self._key(s)
                if key not in self.heads:
                    din, dout = s['in_features'], s['out_features']
                    head = nn.Linear(width, rank * (din + dout))
                    nn.init.normal_(head.weight, std=0.001)
                    nn.init.zeros_(head.bias)
                    with torch.no_grad():
                        head.weight[rank * din:].zero_()
                    self.heads[key] = head

    @staticmethod
    def _key(spec):
        return f"{spec['module']}_{spec['in_features']}_{spec['out_features']}"

    def forward(self, native_vector, model_id):
        if model_id not in self.model_ids:
            raise ValueError('Unknown model identity')
        if native_vector.ndim != 1 or native_vector.numel() != self.project.in_features:
            raise ValueError('Document embedding has the wrong width')
        if not torch.isfinite(native_vector).all():
            raise ValueError('Document embedding is not finite')
        context = F.normalize(self.project(native_vector.float()), dim=-1)
        result = {}
        for spec in self.specs[model_id]:
            identity = self.model_embedding.weight[self.model_ids[model_id]]
            layer = self.layer_embedding.weight[spec['layer']]
            module = self.module_embedding.weight[self.module_names.index(spec['module'])]
            hidden = self.trunk(torch.cat((context, identity, layer, module)))
            values = self.heads[self._key(spec)](hidden)
            din, dout = spec['in_features'], spec['out_features']
            split = self.rank * din
            result[spec['name']] = (values[:split].reshape(self.rank, din),
                                    values[split:].reshape(dout, self.rank))
        return result


def mixed_schedule(count, seed):
    """Shuffle documents and randomly order their two model visits, exactly once each."""
    if count < 1:
        raise ValueError('At least one document is required')
    rng = random.Random(seed)
    indices = list(range(count))
    rng.shuffle(indices)
    for index in indices:
        models = ['3b', '8b']
        rng.shuffle(models)
        for model in models:
            yield index, model


def scan_jsonl(path):
    """Index input rows and pin the complete source without retaining document text."""
    digest, offsets = hashlib.sha256(), []
    with open(path, 'rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            digest.update(line)
            if line.strip():
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f'Invalid JSON at byte {offset}') from exc
                if not isinstance(row, dict) or not isinstance(row.get('text'), str) or not row['text'].strip():
                    raise ValueError(f'Expected a nonempty text field at byte {offset}')
                offsets.append(offset)
    if not offsets:
        raise ValueError('Training JSONL contains no documents')
    return offsets, digest.hexdigest()


def masked_loss(backbone, tokenizer, text, embedding, network, model_id, device, sequence_length, mask_id):
    """Same-position masked-token CE, conditioning the shared generator on this model."""
    from .model import generated_lora
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=sequence_length - 1)
    if not ids or tokenizer.eos_token_id is None:
        raise ValueError('Document requires at least one token and tokenizer EOS')
    clean = torch.tensor([ids + [tokenizer.eos_token_id]], dtype=torch.long, device=device)
    mask = torch.rand(clean.shape, device=device) < torch.rand((), device=device).clamp(min=0.01)
    mask[:, 0] = False
    if not mask.any():
        mask[0, torch.randint(1, clean.shape[1], (), device=device)] = True
    noisy = clean.masked_fill(mask, mask_id)
    factors = network(embedding.to(device), model_id)
    modes = [(module, module.diffusion_lm) for module in backbone.modules()
             if hasattr(module, 'diffusion_lm')]
    try:
        for module, _ in modes:
            module.diffusion_lm = True
        with generated_lora(backbone, factors, network.alpha, network.rank):
            output = backbone(input_ids=noisy, use_cache=False, use_causal_mask=False, return_dict=True)
            logits = output.logits
            if logits.shape[:2] != clean.shape:
                raise ValueError('Nemotron logits do not match input positions')
            loss = F.cross_entropy(logits[mask].float(), clean[mask])
            loss.backward()
    finally:
        for module, mode in modes:
            module.diffusion_lm = mode
    return loss.detach()
