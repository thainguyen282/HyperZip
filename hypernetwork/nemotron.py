"""Document-conditioned hypernetwork training through NVIDIA's native dLLM recipe.

Run with the pinned NeMo environment: python -m hypernetwork.nemotron -c CONFIG.
Only adapter conditioning and model-weight serialization are custom; NeMo owns
corruption, hybrid loss, accumulation, optimization, validation and resume state.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.checkpoint.utils import resolve_restore_from_to_checkpoint_dir
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe
from transformers import AutoConfig

from .model import HyperConfig, HyperNetwork, generated_lora, target_spec

MODEL = "nvidia/Nemotron-Labs-Diffusion-8B-Base"
REVISION = "59ff0ffee284112fc6ccf37493ced41a11031434"
UPSTREAM = "20753eed182d7c2379987dc113d0793dda34b717"
FORMAT = "nemotron_hypernetwork_recipe_v2"


def read_documents(data_dir, sequence_length):
    root = Path(data_dir)
    metadata = json.loads((root / "metadata.json").read_text())
    if (metadata.get("format") != "nemotron_hypernetwork_documents_v1" or
            metadata.get("model") != MODEL or metadata.get("model_revision") != REVISION or
            metadata.get("sequence_length") != sequence_length):
        raise ValueError("Dataset/model/tokenizer mismatch")
    if sequence_length < 1024 or sequence_length % 1024:
        raise ValueError("sequence_length must be a positive multiple of 1024")
    rows = {split: [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()
                    if line.strip()] for split in ("train", "validation")}
    for split, values in rows.items():
        if not values or any(not 2 <= len(r["input_ids"]) <= sequence_length for r in values):
            raise ValueError(f"Missing documents or invalid token lengths: {split}")
    if {r["document_id"] for r in rows["train"]} & {r["document_id"] for r in rows["validation"]}:
        raise ValueError("Training and validation documents overlap")
    digest = hashlib.sha256(b"".join((root / name).read_bytes() for name in
                                    ("metadata.json", "train.jsonl", "validation.jsonl"))).hexdigest()
    return rows, metadata, digest


def load_documents(data_dir, split, seq_length=1024):
    """Adapt existing document JSONL to NeMo's unshifted dataset contract."""
    from datasets import Dataset
    rows, metadata, _ = read_documents(data_dir, seq_length)
    examples = []
    for row in rows[split]:
        ids = row["input_ids"]
        padding = seq_length - len(ids)
        examples.append(dict(input_ids=ids + [metadata["eos_token_id"]] * padding,
                             loss_mask=[0] + [1] * (len(ids) - 1) + [0] * padding,
                             attention_mask=[1] * len(ids) + [0] * padding))
    return Dataset.from_list(examples)


@contextmanager
def checkpoint_layers(base):
    """Keep adapter hooks alive through non-reentrant backward recomputation."""
    from torch.utils.checkpoint import checkpoint
    originals = []
    try:
        for layer in base.encoder.layers:
            original = layer.forward
            def forward(*args, _original=original, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)
            originals.append((layer, original))
            layer.forward = forward
        yield
    finally:
        for layer, original in originals:
            layer.forward = original


class HypernetworkModel(HFCheckpointingMixin, torch.nn.Module):
    """Composite model: trainable generator, immutable pretrained backbone."""
    def __init__(self, backbone, hypernetwork, contract):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.hypernetwork = hypernetwork
        self.config = backbone.config
        self.contract = contract

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, input_ids, labels=None, masked_indices=None, skip_loss=True, **kwargs):
        kwargs.pop("use_cache", None)
        result = self.backbone(input_ids=input_ids, labels=labels, masked_indices=masked_indices,
                               skip_loss=skip_loss, use_cache=False, **kwargs)
        if getattr(result, "causal_logits", None) is None:
            raise RuntimeError("Missing AR branch: block_diff configuration is required")
        return result

    def context(self, clean_ids, loss_mask):
        if clean_ids.shape[0] != 1:
            raise ValueError("Generated adapters require local_batch_size=1")
        # Dataset masks are contiguous; re-include the clean first token for conditioning.
        valid = loss_mask[0].bool().clone()
        valid[0] = True
        ids = clean_ids[0, valid][:self.contract["max_doc_tokens"]]
        with torch.no_grad():
            return self.get_input_embeddings()(ids).float().mean(dim=0)

    def save_pretrained(self, save_directory, checkpointer=None, **kwargs):
        # NeMo publishes this directory only after optimizer/RNG/loader state is saved.
        root = Path(save_directory) / "model"
        root.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.hypernetwork.state_dict().items()},
                  str(root / "hypernetwork.safetensors"))
        (root / "hypernetwork_config.json").write_text(json.dumps(
            {"format": FORMAT, "contract": self.contract}, indent=2) + "\n")

    def load_pretrained(self, checkpoint_dir, checkpointer=None):
        root = Path(checkpoint_dir) / "model"
        config = root / "hypernetwork_config.json"
        if not config.exists():
            raise ValueError("Expected a native hypernetwork recipe checkpoint; legacy checkpoints cannot resume")
        saved = json.loads(config.read_text())
        if saved.get("format") != FORMAT or saved.get("contract") != self.contract:
            raise ValueError("Hypernetwork checkpoint conditioning/data/model contract mismatch")
        self.hypernetwork.load_state_dict(load_file(str(root / "hypernetwork.safetensors")))


def build_model(pretrained_model_name_or_path=MODEL, revision=REVISION, *, data_dir,
                sequence_length=1024, max_doc_tokens=512, width=64, depth=2, rank=4, alpha=4,
                seed=42, trust_remote_code=True):
    if pretrained_model_name_or_path != MODEL or revision != REVISION or max_doc_tokens < 1:
        raise ValueError("Use the pinned Nemotron model and positive conditioning length")
    _, _, digest = read_documents(data_dir, sequence_length)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=trust_remote_code,
                                       dlm_paradigm="block_diff", block_size=32)
    config.use_cache = False
    base = NeMoAutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, config=config,
        trust_remote_code=trust_remote_code, torch_dtype=torch.bfloat16).eval().requires_grad_(False)
    spec = target_spec(base)
    hyper_config = HyperConfig(embedding_dim=base.get_input_embeddings().embedding_dim,
                              hidden_dim=width, depth=depth, rank=rank, alpha=alpha)
    hyper = HyperNetwork(hyper_config, spec)
    contract = dict(model=MODEL, revision=REVISION, upstream=UPSTREAM, data_sha256=digest,
                    hypernetwork=asdict(hyper_config), targets=spec, sequence_length=sequence_length,
                    max_doc_tokens=max_doc_tokens, seed=seed, context="frozen_token_embedding_mean_v1",
                    objective="native_nemotron_hybrid_alpha_0.3")
    return HypernetworkModel(base, hyper, contract)


class HypernetworkDiffusionLMSFTRecipe(DiffusionLMSFTRecipe):
    def setup(self):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Hypernetwork recipe currently supports one GPU/rank only")
        if self.cfg.get("step_scheduler.local_batch_size", 1) != 1:
            raise ValueError("Hypernetwork recipe requires local_batch_size=1; use global_batch_size for accumulation")
        if any(self.cfg.get(f"distributed.{axis}_size", 1) != 1 for axis in ("tp", "cp", "pp")):
            raise ValueError("Hypernetwork recipe requires TP=CP=PP=1")
        if self.cfg.get("distributed.activation_checkpointing", False):
            raise ValueError("The adapter hook owns backbone checkpointing; disable infrastructure activation_checkpointing")
        if self.cfg.get("dllm.mode") != "hybrid" or self.cfg.get("dllm.ar_loss_alpha") != 0.3:
            raise ValueError("Expected the Nemotron hybrid objective with diffusion weight 0.3")
        restore = self.cfg.get("checkpoint.restore_from", None)
        if restore:
            saved = resolve_restore_from_to_checkpoint_dir(self.cfg.checkpoint.checkpoint_dir, restore)
            if saved is None or not (Path(saved) / "model/hypernetwork_config.json").is_file():
                raise ValueError("Expected a native hypernetwork recipe checkpoint; legacy checkpoints cannot resume")
        # The model and both dataset factories must describe the same prepared corpus.
        for name in ("dataset", "validation_dataset"):
            if (Path(self.cfg.get(f"{name}.data_dir")).resolve() != Path(self.cfg.model.data_dir).resolve() or
                    self.cfg.get(f"{name}.seq_length") != self.cfg.model.sequence_length):
                raise ValueError("Model and dataset preparation settings disagree")
        self.cfg.model.seed = self.cfg.get("seed", 42)
        super().setup()
        model = self.model_parts[0]
        expected = {id(p) for p in model.hypernetwork.parameters()}
        actual = {id(p) for opt in self.optimizer for group in opt.param_groups for p in group["params"]}
        if expected != actual or any(p.requires_grad for p in model.backbone.parameters()):
            raise RuntimeError("Optimizer must contain exactly the hypernetwork parameters")
        for opt in self.optimizer:
            opt.register_step_pre_hook(self._check_gradients)
        self._self_cond_base_seed = self.cfg.get("seed", 42)

    @staticmethod
    def _check_gradients(optimizer, args, kwargs):
        grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError("Missing or nonfinite hypernetwork gradients")
        if not any(torch.count_nonzero(g) for g in grads):
            raise RuntimeError("Zero hypernetwork gradients")

    def _forward_backward_step(self, idx, batch, **kwargs):
        model = self.model_parts[0]
        clean = batch.get("_clean_input_ids", batch["input_ids"]).to(self.dist_env.device)
        context = model.context(clean, batch["loss_mask"].to(self.dist_env.device))
        factors = model.hypernetwork(context)
        config = model.hypernetwork.config
        with checkpoint_layers(model.backbone), generated_lora(model.backbone, factors, config.alpha, config.rank):
            super()._forward_backward_step(idx, batch, **kwargs)
        if not all(torch.isfinite(loss).all() for loss in kwargs["loss_buffer"]):
            raise RuntimeError("Nonfinite hybrid loss")


def main():
    cfg = parse_args_and_load_config()
    recipe = HypernetworkDiffusionLMSFTRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
