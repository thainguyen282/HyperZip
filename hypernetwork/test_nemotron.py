"""CPU regression tests exercising NVIDIA's inherited forward/backward method."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.recipes.dllm.strategy import HybridStrategy
from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe
from .model import HyperConfig, HyperNetwork, target_spec
from .nemotron import HypernetworkModel, HypernetworkDiffusionLMSFTRecipe, load_documents


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8)
        self.self_attn.v_proj = nn.Linear(8, 4)
        self.out = nn.Linear(4, 8)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.tanh(self.self_attn.q_proj(x) + self.out(self.self_attn.v_proj(x)))


class ToyHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.encoder = nn.Module()
        self.encoder.embed_tokens = nn.Embedding(128, 8)
        self.encoder.layers = nn.ModuleList([ToyLayer()])
        self.head = nn.Linear(8, 128)
        self.include_ar = True

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def forward(self, input_ids, labels, masked_indices, skip_loss, use_cache):
        assert skip_loss and not use_cache and torch.equal(input_ids, labels)
        self.last_mask = masked_indices.clone()
        noisy = torch.where(masked_indices, 100, input_ids)
        def logits(ids):
            return self.head(self.encoder.layers[0](self.encoder.embed_tokens(ids)))
        return SimpleNamespace(logits=logits(noisy), causal_logits=logits(input_ids) if self.include_ar else None)


def toy_recipe():
    base = ToyHybrid()
    hyper = HyperNetwork(HyperConfig(embedding_dim=8, identity_dim=2, hidden_dim=12,
                                     depth=2, rank=2, alpha=2), target_spec(base))
    model = HypernetworkModel(base, hyper, dict(max_doc_tokens=5, hypernetwork=asdict(hyper.config)))
    recipe = HypernetworkDiffusionLMSFTRecipe(ConfigNode({}))
    recipe.model_parts = [model]
    recipe.mesh_context = MeshContext()
    recipe.dist_env = SimpleNamespace(device="cpu", world_size=1)
    recipe.device_mesh = None
    recipe.distributed_config = SimpleNamespace(autocast_dtype=None)
    recipe.te_fp8 = None
    recipe.dllm_strategy = HybridStrategy()
    recipe.dllm_loss_fn = recipe.dllm_strategy.create_loss_fn({"ar_loss_alpha": 0.3})
    recipe.dllm_eps, recipe.dllm_block_size, recipe.dllm_half_life_ratio = .001, None, None
    recipe.mask_token_id = 100
    recipe.step_scheduler = SimpleNamespace(step=0, grad_acc_steps=1)
    recipe._dllm_loss_buffer = []
    return recipe


class NemotronHypernetworkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.recipe = toy_recipe()
        self.model = self.recipe.model_parts[0]
        self.batch = dict(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 11, 11]]),
                          loss_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 0, 0.]]))

    def backward(self, batch=None):
        batch = deepcopy(self.batch if batch is None else batch)
        noise, supervised = self.recipe.dllm_strategy.pre_step(self.recipe, [batch])
        losses = []
        self.recipe._forward_backward_step(0, batch, loss_buffer=losses,
            num_diffusion_tokens=noise, num_ar_tokens=supervised, num_batches=1)
        return losses[0]

    def test_inherited_backward_updates_only_hypernetwork(self):
        original = DiffusionLMSFTRecipe._forward_backward_step
        base_before = {k: v.clone() for k, v in self.model.backbone.state_dict().items()}
        before = {k: v.clone() for k, v in self.model.hypernetwork.state_dict().items()}
        self.model.train()
        self.assertFalse(self.model.backbone.training)
        optimizer = torch.optim.AdamW(self.model.hypernetwork.parameters(), lr=.02)
        optimizer.register_step_pre_hook(self.recipe._check_gradients)
        with patch.object(DiffusionLMSFTRecipe, '_forward_backward_step', autospec=True,
                          side_effect=original) as inherited:
            for _ in range(3):
                optimizer.zero_grad()
                loss = self.backward()
                self.assertGreater(float(loss), float(self.recipe._dllm_loss_buffer[-1]))
                self.assertGreater(float(self.recipe._dllm_loss_buffer[-1]), 0)
                optimizer.step()
            self.assertEqual(inherited.call_count, 3)
        self.assertTrue(all(p.grad is None for p in self.model.backbone.parameters()))
        self.assertTrue(all(torch.equal(v, base_before[k]) for k, v in self.model.backbone.state_dict().items()))
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in self.model.hypernetwork.state_dict().items()))
        self.assertGreater(self.model.backbone.encoder.layers[0].calls, 6)  # backward recomputation
        self.assertFalse(any(m._forward_hooks for m in self.model.backbone.modules()))

    def test_clean_context_padding_and_exception_cleanup(self):
        context = self.model.context(self.batch['input_ids'], self.batch['loss_mask'])
        self.assertFalse(context.requires_grad)
        expected = self.model.get_input_embeddings()(self.batch['input_ids'][0, :5]).float().mean(0)
        torch.testing.assert_close(context, expected)
        padded = deepcopy(self.batch)
        padded['input_ids'][0, 8:] = 99
        torch.testing.assert_close(context, self.model.context(padded['input_ids'], padded['loss_mask']))
        self.backward()
        self.assertFalse(self.model.backbone.last_mask[0, 0])
        self.assertFalse(self.model.backbone.last_mask[0, 8:].any())
        self.model.backbone.include_ar = False
        with self.assertRaisesRegex(RuntimeError, 'Missing AR'):
            self.backward()
        self.assertFalse(any(m._forward_hooks for m in self.model.backbone.modules()))

    def test_checkpoint_roundtrip_and_contract_rejection(self):
        self.backward()
        torch.optim.AdamW(self.model.hypernetwork.parameters(), lr=.02).step()
        context = self.model.context(self.batch['input_ids'], self.batch['loss_mask'])
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            clone = toy_recipe().model_parts[0]
            clone.load_pretrained(directory)
            for key, values in self.model.hypernetwork(context).items():
                for a, b in zip(values, clone.hypernetwork(context)[key]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            clone.contract['max_doc_tokens'] = 4
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                clone.load_pretrained(directory)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                clone.load_pretrained(Path(directory) / 'legacy')

    def test_masks_and_topology_guards(self):
        dataset = load_documents('data/hypernetwork_fineweb_1367451', 'train')
        self.assertEqual(len(dataset), 32)
        for row in dataset:
            self.assertEqual(len(row['input_ids']), 1024)
            self.assertEqual(sum(row['loss_mask']), sum(row['attention_mask']) - 1)
        with patch.dict('os.environ', WORLD_SIZE='2'):
            with self.assertRaisesRegex(ValueError, 'one GPU'):
                self.recipe.setup()
        with self.assertRaisesRegex(ValueError, 'local_batch_size'):
            self.model.context(self.batch['input_ids'].repeat(2, 1), self.batch['loss_mask'].repeat(2, 1))

    def test_native_config_and_early_legacy_rejection(self):
        cfg = parse_args_and_load_config(argv=['-c', 'attempt/hypernetwork_nemotron.yaml'])
        recipe = HypernetworkDiffusionLMSFTRecipe(cfg)
        self.assertEqual(recipe.cfg.wandb.extra['mode'], 'offline')
        self.assertIsNone(recipe.cfg.step_scheduler.num_epochs)
        with tempfile.TemporaryDirectory() as directory:
            cfg.set_by_dotted('checkpoint.restore_from', directory)
            recipe = HypernetworkDiffusionLMSFTRecipe(cfg)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                recipe.setup()

    def test_lazy_dataloader_resume_preserves_next_document(self):
        from torchdata.stateful_dataloader import StatefulDataLoader
        from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
        from .smoke_nemotron import loader_state
        def loader():
            data = list(range(32))
            sampler = StatefulDistributedSampler(data, seed=42, num_replicas=1, rank=0, shuffle=True)
            return StatefulDataLoader(data, sampler=sampler, batch_size=1, num_workers=0)
        reference = loader()
        iterator = iter(reference)
        for _ in range(5):
            next(iterator)
        saved = deepcopy(reference.state_dict())
        resumed = loader()
        resumed.load_state_dict(deepcopy(saved))
        self.assertEqual(loader_state(resumed), saved)
        torch.testing.assert_close(next(iter(resumed)), next(iterator), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
