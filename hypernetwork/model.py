"""Shared document/layer/module MLP and differentiable generated LoRA.

The backbone is never registered inside the hypernetwork. Generated factors
remain in the autograd graph; no Parameter assignment or .data copies are used.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import re

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class HyperConfig:
    embedding_dim: int = 1536
    identity_dim: int = 32
    hidden_dim: int = 1280
    depth: int = 2
    rank: int = 8
    alpha: float = 8.0

    def __post_init__(self):
        for key in ('embedding_dim', 'identity_dim', 'hidden_dim', 'depth', 'rank'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('alpha must be finite and positive')


def target_spec(model, targets=('q_proj', 'v_proj')):
    """Record exact projection names/shapes, including grouped-query V shapes."""
    result = []
    for name, module in model.named_modules():
        match = re.search(r'(?:^|\.)layers\.(\d+)\..*\.([^\.]+)$', name)
        if match and match[2] in targets:
            if not isinstance(module, nn.Linear):
                raise ValueError(f'Expected an unadapted linear projection: {name}')
            result.append(dict(name=name, layer=int(match[1]), module=match[2],
                               in_features=module.in_features, out_features=module.out_features))
    if not result or {s['module'] for s in result} != set(targets):
        raise ValueError(f'Backbone must contain layer projections {targets}')
    layers = sorted({s['layer'] for s in result})
    if layers != list(range(len(layers))) or len(result) != len(layers) * len(targets):
        raise ValueError('Each contiguous transformer layer must have every target projection')
    return result


class HyperNetwork(nn.Module):
    def __init__(self, config, spec):
        super().__init__()
        self.config, self.spec = config, spec
        if not spec or len({s['name'] for s in spec}) != len(spec):
            raise ValueError('Expected unique target module names')
        self.module_names = sorted({s['module'] for s in spec})
        self.layers = nn.Embedding(max(s['layer'] for s in spec) + 1, config.identity_dim)
        self.modules_embedding = nn.Embedding(len(self.module_names), config.identity_dim)
        sizes = [config.embedding_dim + 2 * config.identity_dim] + [config.hidden_dim] * config.depth
        self.trunk = nn.Sequential(*[item for left, right in zip(sizes, sizes[1:])
                                     for item in (nn.Linear(left, right), nn.SiLU())])
        self.heads = nn.ModuleDict()
        for name in self.module_names:
            shapes = {(s['in_features'], s['out_features']) for s in spec if s['module'] == name}
            if len(shapes) != 1:
                raise ValueError('Projection shapes must be uniform across layers')
            din, dout = shapes.pop()
            head = nn.Linear(config.hidden_dim, config.rank * (din + dout))
            # Nonzero A, zero B starts at the base model without killing B gradients.
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.weight[config.rank * din:].zero_()
            self.heads[name] = head

    def forward(self, embedding):
        """Generate one document's adapter, batching layer evaluations per head."""
        if embedding.shape != (self.config.embedding_dim,) or not torch.isfinite(embedding).all():
            raise ValueError('Expected one finite document embedding of the configured dimension')
        result = {}
        for index, name in enumerate(self.module_names):
            specs = [s for s in self.spec if s['module'] == name]
            layers = torch.tensor([s['layer'] for s in specs], device=embedding.device)
            ids = torch.full_like(layers, index)
            conditioning = torch.cat((embedding.expand(len(specs), -1),
                                      self.layers(layers), self.modules_embedding(ids)), dim=-1)
            values = self.heads[name](self.trunk(conditioning))
            for row, spec in zip(values, specs):
                split = self.config.rank * spec['in_features']
                result[spec['name']] = (row[:split].reshape(self.config.rank, spec['in_features']),
                                       row[split:].reshape(spec['out_features'], self.config.rank))
        return result

    def configuration(self):
        return {'config': asdict(self.config), 'targets': self.spec}


@contextmanager
def generated_lora(model, factors, alpha, rank):
    """Attach factors for a whole forward/backward or compression session."""
    modules = dict(model.named_modules())
    handles = []
    try:
        for name, (a, b) in factors.items():
            module = modules.get(name)
            if (not isinstance(module, nn.Linear) or a.shape != (rank, module.in_features)
                    or b.shape != (module.out_features, rank)):
                raise ValueError(f'Generated adapter does not match projection {name}')
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError('Generated adapter contains nonfinite weights')
            def hook(module, inputs, output, a=a, b=b):
                # FP32 adapters match PEFT's inference policy on BF16 backbones.
                delta = F.linear(F.linear(inputs[0].to(a.dtype), a), b)
                return (output.to(delta.dtype) + delta * (alpha / rank)).to(output.dtype)
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
