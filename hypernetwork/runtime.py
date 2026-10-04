"""Prepare the archived context and regenerate adapters without source access."""
import hashlib
import time

import numpy as np
import torch

from compression.model_loading import configure_runtime
from .checkpoint import (embedding_bytes, generate, load_checkpoint, read_embedding,
                         validate_backbone)
from .embedding import DocumentEncoder
from .model import generated_lora


def prepare(args, config, *, raw=None, archive=None):
    started = time.perf_counter()
    saved = archive.header.get('hypernetwork') if archive else None
    path = args.hypernetwork_path or (saved or {}).get('path')
    if not path:
        return None
    if archive and saved is None:
        raise ValueError('Cannot add a hypernetwork when decoding a base or saved-LoRA archive')
    if config.lora_path:
        raise ValueError('Choose a hypernetwork or a saved LoRA adapter, not both')
    # The document encoder can initialize cuBLAS before the backbone is loaded.
    # Configure deterministic CUDA before either model creates a CUDA handle.
    configure_runtime(config)
    network, metadata, identity = load_checkpoint(path, saved)
    if archive:
        context_data = archive.context
    elif args.context_embedding:
        context_data = embedding_bytes(np.load(args.context_embedding, allow_pickle=False))
    else:
        settings = metadata['encoder']
        if settings.get('pooling') != 'last_token_l2_v1':
            raise ValueError('Unsupported document encoder pooling')
        device = config.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = config.dtype if config.dtype != 'auto' else ('bfloat16' if device.startswith('cuda') else 'float32')
        encoder = DocumentEncoder(settings['model_path'], revision=settings.get('revision'),
            max_length=settings['max_length'], device=device, dtype=dtype,
            trust_remote_code=config.trust_remote_code, local_files_only=config.local_files_only)
        context_data = embedding_bytes(encoder.encode(raw.decode('utf-8')).numpy())
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    context = read_embedding(context_data, network.config.embedding_dim)
    factors, fingerprint = generate(network, context)
    if saved and fingerprint != saved['adapter_sha256']:
        raise ValueError('Regenerated hypernetwork adapter checksum mismatch')
    header = dict(**identity, embedding_dim=network.config.embedding_dim,
                  context_sha256=hashlib.sha256(context_data).hexdigest(),
                  adapter_sha256=fingerprint, generator='cpu_fp32_v1')
    return dict(metadata=metadata, header=header, context=context_data,
                factors=factors, rank=network.config.rank, alpha=network.config.alpha,
                personalization_seconds=time.perf_counter() - started)


def attach(prepared, config, model):
    validate_backbone(prepared['metadata'], config, model)
    device = next(model.parameters()).device
    factors = {name: tuple(value.to(device) for value in pair)
               for name, pair in prepared['factors'].items()}
    return generated_lora(model, factors, prepared['alpha'], prepared['rank'])
