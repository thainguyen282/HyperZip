"""Hypernetwork training entrypoint; Nemotron delegates to NVIDIA's recipe."""
if __package__ in (None, ""):
    # Support both `python hypernetwork/train.py` and `python -m hypernetwork.train`.
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "hypernetwork"
import json
import math
from contextlib import contextmanager
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from config import ModelConfig, get_diffusion_config
from compression.model_loading import load_model
from .checkpoint import save_checkpoint
from .model import HyperConfig, HyperNetwork, generated_lora, target_spec

from .train_config import load_nemotron_config


@contextmanager
def checkpoint_layers(model, enabled=True):
    """Checkpoint decoder layers without enabling backbone dropout/noise logic."""
    from torch.utils.checkpoint import checkpoint
    originals = []
    try:
        if enabled:
            for layer in model.model.layers:
                original = layer.forward
                def forward(*args, _original=original, **kwargs):
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)
                originals.append((layer, original))
                layer.forward = forward
        yield
    finally:
        for layer, original in originals:
            layer.forward = original


def masked_diffusion_loss(model, tokens, *, eos, mask, block_size, sequence_length, rng):
    """Sample a partially masked block after a clean prefix, as in compression.

    Fast-dLLM predicts slot i from logit i-1. The first slot of each block is
    the known head token, so only the remaining slots are denoising targets.
    Keeping the backbone in eval mode avoids its implicit noise injection;
    autograd remains enabled through our explicit corruption and LoRA hooks.
    """
    if not tokens or block_size < 2 or sequence_length < block_size:
        raise ValueError('Need tokens and sequence_length >= block_size >= 2')
    if len(tokens) > sequence_length - 1:
        start = rng.randrange(len(tokens) - sequence_length + 2)
        tokens = tokens[start:start + sequence_length - 1]
    ids = [eos] + tokens
    blocks = [start for start in range(0, len(ids), block_size) if len(ids) - start >= 2]
    start = rng.choice(blocks)
    ids = ids[:start + block_size]
    device = next(model.parameters()).device
    clean = torch.tensor([ids], dtype=torch.long, device=device)
    candidates = torch.arange(start + 1, len(ids), device=device)
    probability = 0.001 + 0.999 * rng.random()
    # Use the local RNG for reproducibility independent of adapter initialization.
    selected = torch.tensor([rng.random() < probability for _ in candidates], device=device)
    if not selected.any():
        selected[rng.randrange(len(candidates))] = True
    positions = candidates[selected]
    noisy = clean.clone()
    noisy[0, positions] = mask
    output = model(input_ids=noisy, use_cache=False, block_size=block_size,
                   use_block_cache=False, logits_to_keep=len(ids) - start, return_dict=True)
    logits = output.logits[0, positions - start - 1].float()
    return F.cross_entropy(logits, clean[0, positions])


def documents(path, rng, buffer_size=128):
    """Cycle a prepared JSONL corpus with a bounded shuffle buffer."""
    while True:
        buffer, count = [], 0
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item['input_ids']:
                    continue
                count += 1
                buffer.append(item)
                if len(buffer) >= buffer_size:
                    yield buffer.pop(rng.randrange(len(buffer)))
        rng.shuffle(buffer)
        yield from buffer
        if count == 0:
            raise ValueError('Prepared corpus has no nonempty documents')


def train(args):
    root = Path(args.data)
    prepared = json.loads((root / 'metadata.json').read_text())
    if prepared.get('format') != 'hyperzip_training_data_v1':
        raise ValueError('Unsupported prepared corpus format')
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.gradient_accumulation < 1 or args.learning_rate <= 0:
        raise ValueError('Steps, gradient accumulation, and learning rate must be positive')
    if args.block_size < 2 or args.sequence_length < args.block_size:
        raise ValueError('Need sequence_length >= block_size >= 2')
    model_path = str(Path(args.model_path).resolve()) if Path(args.model_path).exists() else args.model_path
    if prepared['tokenizer_path'] != model_path:
        raise ValueError('Prepared token IDs belong to a different backbone tokenizer')
    config = ModelConfig(model='fast_dllm', model_path=model_path, revision=prepared.get('tokenizer_revision'),
        device=args.device, dtype=args.dtype, seed=args.seed, trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        diffusion=get_diffusion_config('fast_dllm', attention_block_size=args.block_size,
                                      use_dual_cache=False, confidence_threshold=1.0))
    owner = load_model(config)
    base = owner.model.eval().requires_grad_(False)
    if args.sequence_length > owner.max_length:
        raise ValueError('Training sequence exceeds the backbone context limit')
    hyperconfig = HyperConfig(embedding_dim=prepared['encoder']['embedding_dim'],
        hidden_dim=args.hidden_dim, depth=args.depth, rank=args.rank, alpha=args.alpha)
    network = HyperNetwork(hyperconfig, target_spec(base)).to(config.device).train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate, weight_decay=0.01)
    rng = random.Random(args.seed)
    stream = documents(root / 'documents.jsonl', random.Random(args.seed))
    warmup = max(1, int(args.steps * 0.1))
    history = []
    for step in range(args.steps):
        scale = ((step + 1) / warmup if step < warmup else
                 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))))
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate * scale
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.gradient_accumulation):
            document = next(stream)
            context = torch.tensor(document['embedding'], dtype=torch.float32, device=config.device)
            factors = network(context)
            with generated_lora(base, factors, hyperconfig.alpha, hyperconfig.rank), \
                    checkpoint_layers(base, not args.no_gradient_checkpointing):
                loss = masked_diffusion_loss(base, document['input_ids'], eos=owner.tokenizer.eos_token_id,
                    mask=owner.mask, block_size=args.block_size, sequence_length=args.sequence_length, rng=rng)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss / args.gradient_accumulation).backward()
            total += loss.detach().item() / args.gradient_accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        metrics = dict(step=step + 1, loss=total, learning_rate=optimizer.param_groups[0]['lr'],
                       grad_norm=float(grad_norm))
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
    save_checkpoint(args.output, network, backbone=dict(model='fast_dllm', model_path=config.model_path,
        revision=config.revision), encoder=prepared['encoder'], training=dict(
        steps=args.steps, gradient_accumulation=args.gradient_accumulation, seed=args.seed,
        sequence_length=args.sequence_length, block_size=args.block_size,
        objective='masked_block_clean_prefix_v1', learning_rate=args.learning_rate))
    (Path(args.output) / 'training.json').write_text(json.dumps(history, indent=2) + '\n')

def main(argv=None):
    from .nemotron import HypernetworkDiffusionLMSFTRecipe
    cfg = load_nemotron_config(argv)
    print(cfg, flush=True)
    recipe = HypernetworkDiffusionLMSFTRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == '__main__':
    main()
