#!/usr/bin/env python3
"""Train one identity-conditioned generator on paired 3B/8B document visits."""
import argparse
import json
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

from hypernetwork.embedding import DocumentEncoder
from hypernetwork.model import target_spec
from hypernetwork.multimodel import MultiModelHypernetwork, masked_loss, mixed_schedule, scan_jsonl

MODELS = {
    '3b': ('nvidia/Nemotron-Labs-Diffusion-3B-Base', '4bd78816ba988ca8b71b8f432b00b682572b236c'),
    '8b': ('nvidia/Nemotron-Labs-Diffusion-8B-Base', '59ff0ffee284112fc6ccf37493ced41a11031434'),
}
ENCODER = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
ENCODER_REVISION = 'a9af15a6372d7d6b25e9fb07c2ccb9e1fe645644'


def save(root, network, optimizer, contract, cursor):
    dest = root / f'step_{cursor}'
    dest.mkdir(exist_ok=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in network.state_dict().items()},
              str(dest / 'hypernetwork.safetensors'))
    (dest / 'config.json').write_text(json.dumps({'format': 'hyperzip_multimodel_v1',
        'contract': contract, 'cursor': cursor}, indent=2) + '\n')
    torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(),
                'python_rng': random.getstate()}, dest / 'training_state.pt')
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='Training JSONL: one nonempty text field per document')
    p.add_argument('--output', required=True, type=Path, help='New output directory')
    p.add_argument('--resume', type=Path, help='Trusted step_N checkpoint directory; use a new output directory')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--encoder-device', default=None)
    p.add_argument('--sequence-length', type=int, default=1024)
    p.add_argument('--encoder-max-length', type=int, default=1024)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--alpha', type=float, default=4)
    p.add_argument('--identity-dim', type=int, default=32)
    p.add_argument('--context-dim', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mask-token-id', type=int, default=100)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--max-steps', type=int, help='Stop early for a smoke test; default visits all pairs')
    p.add_argument('--local-files-only', action='store_true')
    args = p.parse_args()
    if min(args.sequence_length - 1, args.encoder_max_length, args.width, args.depth, args.rank,
           args.identity_dim, args.context_dim, args.save_every) < 1 or args.learning_rate <= 0 or args.alpha <= 0:
        p.error('Dimensions and training parameters must be positive')
    if args.max_steps is not None and args.max_steps < 1:
        p.error('--max-steps must be positive')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        p.error('This experiment requires an available CUDA GPU')
    if args.output.exists():
        p.error('Output directory must not exist')
    offsets, source_hash = scan_jsonl(args.input)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    models, tokenizers, specs = {}, {}, {}
    for name, (model, revision) in MODELS.items():
        tokenizers[name] = AutoTokenizer.from_pretrained(model, revision=revision,
            trust_remote_code=True, local_files_only=args.local_files_only)
        models[name] = AutoModel.from_pretrained(model, revision=revision, trust_remote_code=True,
            local_files_only=args.local_files_only, dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval().requires_grad_(False)
        specs[name] = target_spec(models[name])
    encoder = DocumentEncoder(ENCODER, revision=ENCODER_REVISION, max_length=args.encoder_max_length,
        device=args.encoder_device or args.device, dtype='bfloat16', trust_remote_code=True,
        local_files_only=args.local_files_only)
    settings = dict(native_dim=encoder.metadata['embedding_dim'], context_dim=args.context_dim,
                    identity_dim=args.identity_dim, width=args.width, depth=args.depth,
                    rank=args.rank, alpha=args.alpha)
    network = MultiModelHypernetwork(specs, **settings).to(args.device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.98), eps=1e-5, weight_decay=0.1)
    contract = {'models': MODELS, 'targets': specs, 'generator': settings, 'encoder': encoder.metadata,
                'source_sha256': source_hash, 'documents': len(offsets), 'seed': args.seed,
                'sequence_length': args.sequence_length, 'mask_token_id': args.mask_token_id,
                'objective': 'same_position_random_masked_cross_entropy_v1',
                'schedule': 'one_shuffled_document_pass_both_models_v1', 'learning_rate': args.learning_rate}
    # JSON round-trip normalizes tuples in the model list for exact resume comparison.
    contract = json.loads(json.dumps(contract))
    cursor = 0
    if args.resume:
        prior = json.loads((args.resume / 'config.json').read_text())
        if prior.get('format') != 'hyperzip_multimodel_v1' or prior['contract'] != contract:
            raise ValueError('Resume checkpoint contract differs from this run')
        network.load_state_dict(load_file(str(args.resume / 'hypernetwork.safetensors'), device=args.device))
        state = torch.load(args.resume / 'training_state.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        cursor = prior['cursor']
        if not 0 <= cursor <= 2 * len(offsets):
            raise ValueError('Invalid checkpoint cursor')
        if args.max_steps is not None and args.max_steps <= cursor:
            raise ValueError('--max-steps must exceed the restored step')
    args.output.mkdir(parents=True, exist_ok=False)
    stop = {'requested': False}
    def request_stop(*_):
        stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    started = time.monotonic()
    last_saved = None
    with args.input.open('rb') as source, (args.output / 'training.jsonl').open('a') as log:
        cached_index = None
        for index, (row_index, model_id) in enumerate(mixed_schedule(len(offsets), args.seed)):
            if index < cursor:
                continue
            if row_index != cached_index:
                source.seek(offsets[row_index])
                text = json.loads(source.readline())['text']
                vector = encoder.encode(text)
                cached_index = row_index
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(models[model_id], tokenizers[model_id], text, vector, network,
                               model_id, args.device, args.sequence_length, args.mask_token_id)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            cursor = index + 1
            log.write(json.dumps({'step': cursor, 'document_index': row_index, 'model': model_id,
                                  'loss': float(loss), 'gradient_norm': float(norm)}) + '\n')
            log.flush()
            if cursor % args.save_every == 0 or stop['requested'] or (args.max_steps and cursor >= args.max_steps):
                last_saved = save(args.output, network, optimizer, contract, cursor)
            if stop['requested'] or (args.max_steps and cursor >= args.max_steps):
                break
    if last_saved is None or last_saved.name != f'step_{cursor}':
        last_saved = save(args.output, network, optimizer, contract, cursor)
    status = {'steps': cursor, 'total_steps': 2 * len(offsets), 'completed_one_mixed_epoch': cursor == 2 * len(offsets),
              'checkpoint': str(last_saved.resolve()), 'seconds': time.monotonic() - started}
    (args.output / 'run-status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
