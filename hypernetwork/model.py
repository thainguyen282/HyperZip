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
    architecture: str = 'mlp'
    native_embedding_dim: int | None = None
    projection: str = 'none'

    def __post_init__(self):
        for key in ('embedding_dim', 'identity_dim', 'hidden_dim', 'depth', 'rank'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        if self.architecture not in ('mlp', 'mlp_s', 'mlp_m', 'mlp_l',
                                     'residual_mlp', 'attention', 'perceiver'):
            raise ValueError('Unsupported hypernetwork architecture')
        if self.projection not in ('none', 'linear', 'truncate_l2'):
            raise ValueError('Unsupported context projection')
        if self.projection in ('linear', 'truncate_l2'):
            if type(self.native_embedding_dim) is not int or self.native_embedding_dim < 1:
                raise ValueError('Projection requires a positive native_embedding_dim')
            if self.embedding_dim > self.native_embedding_dim:
                raise ValueError('Projected context cannot exceed native embedding width')
        elif self.native_embedding_dim is not None:
            raise ValueError('native_embedding_dim requires linear or truncate_l2 projection')


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, value):
        return value + self.net(value) / math.sqrt(2)


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


def capture_pre_layer_states(model, spec, input_ids, **forward_kwargs):
    """Read clean, frozen token activations entering each adapted layer."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError('Expected one nonempty token sequence')
    modules = dict(model.named_modules())
    layer_paths = {}
    for item in spec:
        marker = re.search(r'^(.*?layers\.\d+)\.', item['name'])
        if marker is None:
            raise ValueError(f'Cannot find transformer layer for {item["name"]}')
        layer_paths[item['layer']] = marker.group(1)
    states = {}
    handles = []
    try:
        for index, path in layer_paths.items():
            layer = modules.get(path)
            if layer is None:
                raise ValueError(f'Missing transformer layer: {path}')
            def capture(_module, inputs, kwargs, _index=index):
                value = inputs[0] if inputs else kwargs.get('hidden_states')
                if value is None:
                    raise ValueError('Transformer layer did not receive hidden states')
                if value.ndim == 3 and value.shape[0] == 1:
                    value = value[0]
                if value.ndim != 2:
                    raise ValueError('Expected token-by-hidden layer activations')
                states[_index] = value.detach().float()
            handles.append(layer.register_forward_pre_hook(capture, with_kwargs=True))
        with torch.no_grad():    
            model(input_ids=input_ids, **forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if sorted(states) != sorted(layer_paths):
        raise ValueError('Frozen backbone did not execute every adapted layer')
    return [states[index] for index in sorted(states)]


class PerceiverCrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, latent, state):
        key = self.key(state)
        value = self.value(state)
        scores = self.query(latent) @ key.T / self.scale
        return self.output(torch.softmax(scores, dim=-1) @ value)


class PerceiverHyperNetwork(nn.Module):
    """Cross-attend rank-sized queries to each frozen pre-layer token sequence."""
    def __init__(self, config, spec):
        super().__init__()
        if config.architecture != 'perceiver' or config.projection != 'none':
            raise ValueError('Perceiver requires token states and no vector projection')
        self.config, self.spec = config, spec
        if not spec or len({s['name'] for s in spec}) != len(spec):
            raise ValueError('Expected unique target module names')
        self.module_names = sorted({s['module'] for s in spec})
        layers = sorted({s['layer'] for s in spec})
        if layers != list(range(len(layers))) or len(spec) != len(layers) * len(self.module_names):
            raise ValueError('Expected contiguous adapted layers')
        self.num_layers = len(layers)
        self.context_dim = self.num_layers * config.rank * config.hidden_dim
        self.queries = nn.Parameter(torch.randn(config.rank, config.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(PerceiverCrossAttention(config.embedding_dim, config.hidden_dim)
                                    for _ in range(config.depth))
        self.heads_a = nn.ModuleList()
        self.heads_b = nn.ModuleList()
        for item in spec:
            self.heads_a.append(nn.Linear(config.hidden_dim, item['in_features']))
            head_b = nn.Linear(config.hidden_dim, item['out_features'])
            nn.init.zeros_(head_b.weight)
            nn.init.zeros_(head_b.bias)
            self.heads_b.append(head_b)

    def encode_hidden_states(self, states):
        if len(states) != self.num_layers:
            raise ValueError('Expected one token activation sequence per layer')
        latents = []
        for state in states:
            if (state.ndim != 2 or state.shape[0] < 1 or
                    state.shape[1] != self.config.embedding_dim or not torch.isfinite(state).all()):
                raise ValueError('Invalid pre-layer token activations')
            latent = self.queries
            for block in self.blocks:
                latent = block(latent, state.float())
            latents.append(latent)
        return torch.stack(latents).reshape(-1)

    def forward(self, context):
        if isinstance(context, (list, tuple)):
            context = self.encode_hidden_states(context)
        if context.shape != (self.context_dim,) or not torch.isfinite(context).all():
            raise ValueError('Expected one finite Perceiver latent context')
        latents = context.reshape(self.num_layers, self.config.rank, self.config.hidden_dim)
        result = {}
        for index, item in enumerate(self.spec):
            hidden = latents[item['layer']]
            a = self.heads_a[index](hidden)
            b = self.heads_b[index](hidden).T.contiguous()
            result[item['name']] = (a, b)
        return result

    def configuration(self):
        return {'config': asdict(self.config), 'targets': self.spec}


def build_hypernetwork(config, spec):
    if config.architecture == 'perceiver':
        return PerceiverHyperNetwork(config, spec)
    if config.architecture in ('mlp_s', 'mlp_m', 'mlp_l'):
        return GranularMLPHyperNetwork(config, spec)
    return HyperNetwork(config, spec)


class HyperNetwork(nn.Module):
    def __init__(self, config, spec):
        super().__init__()
        self.config, self.spec = config, spec
        if config.architecture == 'perceiver':
            raise ValueError('Use PerceiverHyperNetwork for token-state conditioning')
        self.context_projection = (nn.Linear(config.native_embedding_dim, config.embedding_dim)
                                   if config.projection == 'linear' else None)
        if not spec or len({s['name'] for s in spec}) != len(spec):
            raise ValueError('Expected unique target module names')
        self.module_names = sorted({s['module'] for s in spec})
        self.context_dim = config.embedding_dim
        self.layers = nn.Embedding(max(s['layer'] for s in spec) + 1, config.identity_dim)
        self.modules_embedding = nn.Embedding(len(self.module_names), config.identity_dim)
        sizes = [config.embedding_dim + 2 * config.identity_dim] + [config.hidden_dim] * config.depth
        if config.architecture == 'mlp':
            self.trunk = nn.Sequential(*[item for left, right in zip(sizes, sizes[1:])
                                         for item in (nn.Linear(left, right), nn.SiLU())])
        elif config.architecture == 'residual_mlp':
            self.trunk = nn.Sequential(nn.Linear(sizes[0], config.hidden_dim), nn.SiLU(),
                                       *(ResidualBlock(config.hidden_dim) for _ in range(config.depth - 1)))
        else:
            heads = 4 if config.hidden_dim % 4 == 0 else 1
            self.trunk = nn.Sequential(nn.Linear(sizes[0], config.hidden_dim),
                nn.TransformerEncoder(nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim, nhead=heads,
                    dim_feedforward=4 * config.hidden_dim, dropout=0.0,
                    activation='gelu', batch_first=True, norm_first=True),
                    num_layers=config.depth, enable_nested_tensor=False))
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

    def project_embedding(self, embedding):
        """Map a native encoder vector to the archived context vector."""
        expected = (self.config.native_embedding_dim if self.config.projection in ('linear', 'truncate_l2')
                    else self.config.embedding_dim)
        if embedding.shape != (expected,) or not torch.isfinite(embedding).all():
            raise ValueError('Expected one finite native embedding of the configured dimension')
        if self.context_projection is None:
            if self.config.projection == 'truncate_l2':
                return F.normalize(embedding[:self.config.embedding_dim].float(), dim=-1)
            return embedding
        return F.normalize(self.context_projection(embedding.float()), dim=-1)

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
            if self.config.architecture == 'attention':
                hidden = self.trunk(conditioning.unsqueeze(0))[0]
            else:
                hidden = self.trunk(conditioning)
            values = self.heads[name](hidden)
            for row, spec in zip(values, specs):
                split = self.config.rank * spec['in_features']
                result[spec['name']] = (row[:split].reshape(self.config.rank, spec['in_features']),
                                       row[split:].reshape(spec['out_features'], self.config.rank))
        return result

    def configuration(self):
        return {'config': asdict(self.config), 'targets': self.spec}


class GranularMLPHyperNetwork(nn.Module):
    """Generate both factors, one factor, or one rank row per MLP call.

    L emits [A, B] together, M emits A/B separately, and S emits one rank
    row/column of A/B at a time. All three share task, depth and module
    conditioning; M/S additionally use A/B identity and S uses rank identity.
    """

    def __init__(self, config, spec):
        super().__init__()
        if config.architecture not in ('mlp_s', 'mlp_m', 'mlp_l'):
            raise ValueError('Expected an MLP-S/M/L architecture')
        if not spec or len({item['name'] for item in spec}) != len(spec):
            raise ValueError('Expected unique target module names')
        self.config, self.spec = config, spec
        self.context_dim = config.embedding_dim
        self.module_names = sorted({item['module'] for item in spec})
        layers = sorted({item['layer'] for item in spec})
        if layers != list(range(len(layers))) or len(spec) != len(layers) * len(self.module_names):
            raise ValueError('Expected every target module in each contiguous layer')
        self.context_projection = (nn.Linear(config.native_embedding_dim, config.embedding_dim)
                                   if config.projection == 'linear' else None)
        hidden = config.hidden_dim
        self.task_encoder = nn.Sequential(nn.Linear(config.embedding_dim, hidden), nn.SiLU())
        self.layer_embedding = nn.Embedding(len(layers), hidden)
        self.module_embedding = nn.Embedding(len(self.module_names), hidden)
        self.before_factor = nn.Sequential(*(
            item for _ in range(config.depth) for item in (nn.Linear(hidden, hidden), nn.SiLU())))
        self.after_factor = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.after_rank = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        if config.architecture in ('mlp_s', 'mlp_m'):
            self.factor_embedding = nn.Embedding(2, hidden)
        if config.architecture == 'mlp_s':
            self.rank_embedding = nn.Embedding(config.rank, hidden)
        self.heads = nn.ModuleDict()
        for name in self.module_names:
            shapes = {(item['in_features'], item['out_features']) for item in spec
                      if item['module'] == name}
            if len(shapes) != 1:
                raise ValueError('Projection shapes must be uniform across layers')
            din, dout = shapes.pop()
            if config.architecture == 'mlp_l':
                head = nn.Linear(hidden, config.rank * (din + dout))
                nn.init.normal_(head.weight, std=0.001)
                nn.init.zeros_(head.bias)
                with torch.no_grad():
                    head.weight[config.rank * din:].zero_()
                self.heads[name] = head
            else:
                multiplier = config.rank if config.architecture == 'mlp_m' else 1
                for factor, dimension in (('a', din), ('b', dout)):
                    head = nn.Linear(hidden, multiplier * dimension)
                    if factor == 'b':
                        nn.init.zeros_(head.weight)
                    else:
                        nn.init.normal_(head.weight, std=0.001)
                    nn.init.zeros_(head.bias)
                    self.heads[f'{name}_{factor}'] = head

    project_embedding = HyperNetwork.project_embedding

    def forward(self, embedding):
        if embedding.shape != (self.context_dim,) or not torch.isfinite(embedding).all():
            raise ValueError('Expected one finite document embedding')
        task = self.task_encoder(embedding)
        result = {}
        rank = self.config.rank
        for module_index, name in enumerate(self.module_names):
            specs = [item for item in self.spec if item['module'] == name]
            layer_ids = torch.tensor([item['layer'] for item in specs], device=embedding.device)
            module_ids = torch.full_like(layer_ids, module_index)
            hidden = self.before_factor(task + self.layer_embedding(layer_ids)
                                        + self.module_embedding(module_ids))
            din, dout = specs[0]['in_features'], specs[0]['out_features']
            if self.config.architecture == 'mlp_l':
                combined = self.heads[name](self.after_rank(self.after_factor(hidden)))
                a = combined[:, :rank * din].reshape(len(specs), rank, din)
                b = combined[:, rank * din:].reshape(len(specs), dout, rank)
            else:
                factors = []
                for factor_index, (factor, dimension) in enumerate((('a', din), ('b', dout))):
                    conditioned = self.after_factor(hidden + self.factor_embedding.weight[factor_index])
                    if self.config.architecture == 'mlp_s':
                        ranked = self.after_rank(conditioned[:, None, :]
                                                 + self.rank_embedding.weight[None, :, :])
                        values = self.heads[f'{name}_{factor}'](ranked)
                    else:
                        values = self.heads[f'{name}_{factor}'](self.after_rank(conditioned))
                        values = values.reshape(len(specs), rank, dimension)
                    factors.append(values)
                a, b = factors[0], factors[1].transpose(1, 2).contiguous()
            for index, item in enumerate(specs):
                result[item['name']] = (a[index], b[index])
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
