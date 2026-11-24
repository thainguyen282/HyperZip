"""Two independently conditioned adapter generators with a shared projection."""

import torch
from torch import nn

from .model import generated_lora, target_spec


class CrossModelHypernetworkModel(nn.Module):
    """Pair frozen 3B/8B backbones with separate 512-context hypernetworks.

    Pass mappings keyed by ``3b`` and ``8b``. Hypernetworks must use the
    existing HyperConfig interface and have no internal context projection.
    Forward applies the selected generated adapters for that forward pass.
    This wrapper does not implement activation checkpointing or a training loop.
    """

    MODEL_IDS = ('3b', '8b')

    def __init__(self, backbones, hypernetworks, native_dim):
        super().__init__()
        if set(backbones) != set(self.MODEL_IDS) or set(hypernetworks) != set(self.MODEL_IDS):
            raise ValueError('Expected exactly the 3b and 8b models')
        if type(native_dim) is not int or native_dim < 1:
            raise ValueError('native_dim must be a positive integer')
        if backbones['3b'] is backbones['8b'] or hypernetworks['3b'] is hypernetworks['8b']:
            raise ValueError('Backbones and hypernetworks must be separate instances')
        if {id(p) for p in hypernetworks['3b'].parameters()} & {id(p) for p in hypernetworks['8b'].parameters()}:
            raise ValueError('Hypernetworks must not share parameters')
        for model_id in self.MODEL_IDS:
            hyper = hypernetworks[model_id]
            if (getattr(hyper, 'context_dim', None) != 512 or
                    getattr(hyper.config, 'projection', None) != 'none' or
                    getattr(hyper.config, 'architecture', None) == 'perceiver'):
                raise ValueError('Each hypernetwork must accept a 512-dimensional vector without a projection')
            targets = tuple(sorted({item['module'] for item in hyper.spec}))
            if target_spec(backbones[model_id], targets) != hyper.spec:
                raise ValueError(f'Hypernetwork targets do not match {model_id} backbone')
        self.backbones = nn.ModuleList(backbones[key].eval().requires_grad_(False) for key in self.MODEL_IDS)
        self.hypernetworks = nn.ModuleList(hypernetworks[key] for key in self.MODEL_IDS)
        self.context_projection = nn.Linear(native_dim, 512)

    def _index(self, model_id):
        if model_id not in self.MODEL_IDS:
            raise ValueError(f'Unknown model identity: {model_id!r}')
        return self.MODEL_IDS.index(model_id)

    def train(self, mode=True):
        super().train(mode)
        for backbone in self.backbones:
            backbone.eval()
        return self

    def context(self, model_id, embedding):
        """Project one finite native document embedding for either model."""
        self._index(model_id)
        if (embedding.ndim != 1 or embedding.numel() != self.context_projection.in_features
                or not torch.isfinite(embedding).all()):
            raise ValueError('Expected one finite native document embedding')
        return self.context_projection(embedding.to(dtype=self.context_projection.weight.dtype))

    def generate_adapters(self, model_id, embedding):
        index = self._index(model_id)
        return self.hypernetworks[index](self.context(model_id, embedding))

    def forward(self, model_id, input_ids, embedding, **kwargs):
        index = self._index(model_id)
        hyper = self.hypernetworks[index]
        factors = self.generate_adapters(model_id, embedding)
        kwargs.pop('use_cache', None)
        with generated_lora(self.backbones[index], factors, hyper.config.alpha, hyper.config.rank):
            return self.backbones[index](input_ids=input_ids, use_cache=False, **kwargs)
