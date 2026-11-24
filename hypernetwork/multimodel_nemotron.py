"""Adapters for using the shared multimodel generator with the Nemotron recipe.

This module does not provide a joint 3B/8B scheduler. Both bound adapters must
reference the same MultiModelHypernetwork object and one shared optimizer.
"""
from types import SimpleNamespace

import torch
from torch import nn

from .multimodel import MultiModelHypernetwork


class BoundMultiModelHypernetwork(nn.Module):
    """Bind a model identity while retaining the shared generator's parameters."""
    def __init__(self, shared: MultiModelHypernetwork, model_id: str):
        super().__init__()
        if model_id not in shared.specs:
            raise ValueError(f'No projection specification for model {model_id!r}')
        self.shared = shared
        self.model_id = model_id
        # HypernetworkDiffusionL MSFTRecipe reads this to scale generated factors.
        self.config = SimpleNamespace(rank=shared.rank, alpha=shared.alpha, architecture='mlp')

    def project_embedding(self, embedding):
        # MultiModelHypernetwork projects and normalizes internally; projecting
        # here as well would apply its learned projection twice.
        if embedding.ndim != 1 or embedding.numel() != self.shared.project.in_features:
            raise ValueError('Expected one native document-encoder vector')
        if not torch.isfinite(embedding).all():
            raise ValueError('Document embedding must be finite')
        return embedding.float()

    def forward(self, embedding):
        return self.shared(self.project_embedding(embedding), self.model_id)


def bind_shared_generator(shared, specs, model_id):
    """Bind only when live target names and shapes exactly match the contract."""
    if model_id not in shared.specs or shared.specs[model_id] != specs:
        raise ValueError('Backbone projection specifications do not match the shared generator contract')
    return BoundMultiModelHypernetwork(shared, model_id)


def install_shared_generator(recipe_model, shared, model_id):
    """Install before optimizer construction on a native HypernetworkModel.

    Callers must maintain one optimizer for ``shared`` across all model visits.
    A recipe's independent optimizer/checkpoint loop is not a joint trainer.
    """
    from .model import target_spec
    if not hasattr(recipe_model, 'backbone') or not hasattr(recipe_model, 'hypernetwork'):
        raise TypeError('Expected a native Nemotron HypernetworkModel')
    if recipe_model.contract.get('conditioning_model') != 'document_encoder':
        raise ValueError('The shared multimodel generator requires prepared document-encoder vectors')
    if any(p.requires_grad for p in recipe_model.backbone.parameters()):
        raise ValueError('The Nemotron backbone must be frozen')
    specs = target_spec(recipe_model.backbone)
    recipe_model.hypernetwork = bind_shared_generator(shared, specs, model_id)
    return recipe_model
