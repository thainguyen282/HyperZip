"""Joint 3B/8B hypernetwork training using NeMo's diffusion SFT recipe."""
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.dllm.collate import DLLMCollator
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs
from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe

from hypernetwork.cross_model import CrossModelHypernetworkModel
from hypernetwork.model import HyperConfig, build_hypernetwork, generated_lora, target_spec
from hypernetwork.nemotron import checkpoint_layers
from hypernetwork.scripts.multimodel_exp.data import PairedCollator
from hypernetwork.scripts.multimodel_exp.prepare import MODELS


class JointModel(HFCheckpointingMixin, CrossModelHypernetworkModel):
    """Expose one selected backbone to NeMo; the recipe manages adapter hooks."""
    def __init__(self, backbones, hypernetworks, native_dim, contract):
        super().__init__(backbones, hypernetworks, native_dim)
        self.contract = contract
        self.config = backbones['3b'].config
        self.active_model = '3b'

    def get_input_embeddings(self):
        return self.backbones[0].get_input_embeddings()

    def forward(self, input_ids, labels=None, masked_indices=None, skip_loss=True, **kwargs):
        kwargs.pop('use_cache', None)
        backbone = self.backbones[self._index(self.active_model)]
        inputs = dict(input_ids=input_ids, labels=labels, masked_indices=masked_indices,
                      skip_loss=skip_loss, use_cache=False, **kwargs)
        result = backbone(**filter_forward_kwargs(backbone, inputs))
        if getattr(result, 'causal_logits', None) is None:
            raise RuntimeError('Nemotron block_diff must return causal logits')
        return result

    def save_pretrained(self, save_directory, checkpointer=None, **kwargs):
        root = Path(save_directory) / 'model'
        root.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()
                 if key.startswith(('hypernetworks.', 'context_projection.'))}
        save_file(state, str(root / 'hypernetwork.safetensors'))
        (root / 'hypernetwork_config.json').write_text(json.dumps(self.contract, indent=2) + '\n')

    def load_pretrained(self, checkpoint_dir, checkpointer=None):
        root = Path(checkpoint_dir) / 'model'
        if json.loads((root / 'hypernetwork_config.json').read_text()) != self.contract:
            raise ValueError('Cross-model checkpoint contract mismatch')
        state = load_file(str(root / 'hypernetwork.safetensors'))
        expected = {key for key in self.state_dict() if key.startswith(('hypernetworks.', 'context_projection.'))}
        if set(state) != expected:
            raise ValueError('Cross-model checkpoint is missing trainable weights')
        self.load_state_dict(state, strict=False)


def data_contract(data_dir):
    root = Path(data_dir)
    manifest = json.loads((root / 'metadata.json').read_text())
    if manifest.get('format') != 'cross_model_tulu_v1':
        raise ValueError('Expected prepared cross-model Tulu data')
    if manifest['models'] != {key: list(value) for key, value in MODELS.items()}:
        raise ValueError('Prepared data uses different model revisions')
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return manifest, digest.hexdigest()


def build_model(data_dir, sequence_length=1024, width=64, depth=2, rank=4, alpha=4):
    manifest, digest = data_contract(data_dir)
    if manifest['sequence_length'] != sequence_length:
        raise ValueError('Prepared sequence length differs from model configuration')
    config = HyperConfig(embedding_dim=512, hidden_dim=width, depth=depth, rank=rank, alpha=alpha)
    backbones, hypernetworks = {}, {}
    for model_id, (name, revision) in MODELS.items():
        model_config = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=True,
                                                   dlm_paradigm='block_diff', block_size=32)
        model_config.use_cache = False
        backbone = NeMoAutoModelForCausalLM.from_pretrained(
            name, revision=revision, config=model_config, trust_remote_code=True,
            torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        backbones[model_id] = backbone
        hypernetworks[model_id] = build_hypernetwork(config, target_spec(backbone))
    contract = {'format': 'cross_model_nemotron_v1', 'data_sha256': digest,
                'data': manifest, 'hypernetwork': vars(config)}
    return JointModel(backbones, hypernetworks, manifest['encoder']['embedding_dim'], contract)


class CrossModelDiffusionLMSFTRecipe(DiffusionLMSFTRecipe):
    """Use NeMo corruption and hybrid loss once per model, with one optimizer step."""
    def setup(self):
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('Cross-model training supports one GPU/rank')
        if self.cfg.get('step_scheduler.local_batch_size', 1) != 1:
            raise ValueError('Cross-model training requires local_batch_size=1')
        if any(self.cfg.get(f'distributed.{axis}_size', 1) != 1 for axis in ('tp', 'cp', 'pp')):
            raise ValueError('Cross-model training requires TP=CP=PP=1')
        if self.cfg.get('distributed.activation_checkpointing', False):
            raise ValueError('Disable NeMo activation checkpointing; adapter hooks manage it')
        if self.cfg.get('dllm.mode') != 'hybrid':
            raise ValueError('Cross-model training requires NeMo hybrid loss')
        for dataset in ('dataset', 'validation_dataset'):
            if Path(self.cfg.get(f'{dataset}.data_dir')).resolve() != Path(self.cfg.model.data_dir).resolve():
                raise ValueError('Model and dataset must use the same prepared data')
        super().setup()
        model = self.model_parts[0]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        allowed = {id(p) for p in model.hypernetworks.parameters()} | {id(p) for p in model.context_projection.parameters()}
        actual = {id(p) for opt in self.optimizer for group in opt.param_groups for p in group['params']}
        if expected != allowed or actual != allowed or any(p.requires_grad for p in model.backbones.parameters()):
            raise RuntimeError('Optimizer must contain exactly the two hypernetworks and shared projection')
        for optimizer in self.optimizer:
            optimizer.register_step_pre_hook(self._check_gradients)
        self._self_cond_base_seed = self.cfg.get('seed', 42)

    def _check_gradients(self, optimizer, args, kwargs):
        model = self.model_parts[0]
        for component in (*model.hypernetworks, model.context_projection):
            gradients = [p.grad for p in component.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
                raise RuntimeError('Missing or nonfinite cross-model gradients')

    def _wrap_dataloader_collate(self):
        collators = {}
        for model_id, (name, revision) in MODELS.items():
            tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
            collators[model_id] = DLLMCollator(
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                eos_token_id=tokenizer.eos_token_id, block_size=self.dllm_pad_block_size,
                pad_seq_len_divisible=self.dllm_pad_seq_len_divisible,
                max_seq_len=self.cfg.model.sequence_length,
                supervise_padding=bool(self.cfg.get('dllm.supervise_padding', False)))
        self.dataloader.collate_fn = PairedCollator(collators)
        for loader in self.val_dataloaders.values():
            loader.collate_fn = PairedCollator(collators)

    @staticmethod
    def _flatten(batches):
        result = []
        for pair in batches:
            for model_id in ('3b', '8b'):
                batch = dict(pair[model_id])
                batch['_cross_model_id'] = model_id
                batch['_cross_embedding'] = pair['context_embedding']
                result.append(batch)
        return result

    def _run_train_optim_step(self, batches, max_grad_norm=None):
        return super()._run_train_optim_step(self._flatten(batches), max_grad_norm=max_grad_norm)

    def _apply_corruption(self, input_ids, loss_mask, microbatch_idx=0):
        # NeMo seeds by microbatch index and accumulation count. Each document
        # has two visits, so count both to avoid reusing seeds across steps.
        steps = self.step_scheduler.grad_acc_steps
        self.step_scheduler.grad_acc_steps = 2 * steps
        try:
            return super()._apply_corruption(input_ids, loss_mask, microbatch_idx)
        finally:
            self.step_scheduler.grad_acc_steps = steps

    def _compute_loss_denominators(self, batches, num_noise_tokens, num_supervised_tokens):
        # The parent calls this after its native corruption pre-step.
        self._cross_counts = {}
        for model_id in ('3b', '8b'):
            selected = [batch for batch in batches if batch['_cross_model_id'] == model_id]
            if selected:
                self._cross_counts[model_id] = (
                    sum(int(b['_noise_mask'].sum()) for b in selected),
                    sum(int(b['loss_mask'].sum()) for b in selected))
        return num_noise_tokens, num_supervised_tokens

    def _forward_backward_step(self, idx, batch, **kwargs):
        model_id = batch.pop('_cross_model_id')
        embedding = batch.pop('_cross_embedding').to(self.dist_env.device)
        model = self.model_parts[0]
        model.active_model = model_id
        index = model._index(model_id)
        hyper = model.hypernetworks[index]
        factors = hyper(model.context(model_id, embedding))
        noise, supervised = self._cross_counts[model_id]
        # Doubling both denominators is equivalent to averaging the two model losses.
        # max(..., 1) matches the native loss's zero-denominator policy.
        kwargs['num_diffusion_tokens'] = 2 * max(noise, 1)
        kwargs['num_ar_tokens'] = 2 * max(supervised, 1)
        with checkpoint_layers(model.backbones[index]), generated_lora(
                model.backbones[index], factors, hyper.config.alpha, hyper.config.rank):
            super()._forward_backward_step(idx, batch, **kwargs)
        if not torch.isfinite(kwargs['loss_buffer'][-1]):
            raise RuntimeError(f'Nonfinite {model_id} loss')

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        # Equal weight per model, with each branch using its own native token counts.
        totals = {'3b': 0.0, '8b': 0.0}
        counts = {'3b': 0, '8b': 0}
        supervised_tokens = 0
        with ScopedRNG(seed=1, ranked=True):
            for model in self.model_parts:
                model.eval()
            for pair in val_dataloader:
                for batch in self._flatten([pair]):
                    model_id = batch['_cross_model_id']
                    noise, supervised = self.dllm_strategy.pre_step(self, [batch])
                    supervised_tokens += supervised
                    self._compute_loss_denominators([batch], noise, supervised)
                    losses = []
                    self.dllm_strategy.forward_backward(self, 0, batch, loss_buffer=losses,
                        num_diffusion_tokens=noise, num_ar_tokens=supervised, num_batches=1, is_train=False)
                    totals[model_id] += 2 * losses[0].item()
                    counts[model_id] += 1
        self._dllm_loss_buffer.clear()
        self._dflash_correct_per_pos_buffer.clear()
        self._dflash_count_per_pos_buffer.clear()
        if not all(counts.values()):
            raise ValueError('Validation requires examples for both models')
        value = sum(totals[k] / counts[k] for k in totals) / 2
        return MetricsSample(step=self.step_scheduler.step, epoch=self.step_scheduler.epoch,
                             metrics={'val_loss': value, 'lr': self.optimizer[0].param_groups[0]['lr'],
                                      'num_label_tokens': supervised_tokens,
                                      'mem': torch.cuda.max_memory_allocated() / 1024**3})


def main():
    recipe = CrossModelDiffusionLMSFTRecipe(parse_args_and_load_config())
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == '__main__':
    main()
