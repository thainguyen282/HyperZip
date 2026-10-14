"""Prepare the archived context and regenerate adapters without source access."""
import hashlib
import time

import numpy as np
import torch

from compression.model_loading import configure_runtime
from .checkpoint import (NEMOTRON_FORMAT, embedding_bytes, generate, load_checkpoint, read_embedding,
                         validate_backbone)
from .embedding import DocumentEncoder, project_context
from .model import capture_pre_layer_states, generated_lora


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
    if metadata['format'] == NEMOTRON_FORMAT or network.config.architecture == 'perceiver':
        return dict(network=network, metadata=metadata, identity=identity, saved=saved,
                    context_data=(archive.context if archive else
                                  embedding_bytes(np.load(args.context_embedding, allow_pickle=False))
                                  if args.context_embedding else None), started=started)
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
        projection = settings.get('projection', 'identity_v1')
        if projection not in ('identity_v1', 'adaptive_mean_l2_v1', 'learned_linear_l2_v1'):
            raise ValueError('Unsupported context projection')
        if (projection == 'learned_linear_l2_v1') != (network.config.projection == 'linear'):
            raise ValueError('Checkpoint and encoder projection mismatch')
        if ('native_embedding_dim' in settings and
                encoder.metadata['embedding_dim'] != settings['native_embedding_dim']):
            raise ValueError('Document encoder output width mismatch')
        native = encoder.encode(raw.decode('utf-8'))
        if projection == 'learned_linear_l2_v1':
            with torch.no_grad():
                context = network.cpu().float().eval().project_embedding(native.float())
        else:
            context = project_context(native, network.config.embedding_dim)
        context_data = embedding_bytes(context.detach().cpu().numpy())
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


def finish(prepared, config, model, tokenizer, *, raw=None):
    """Finish conditioning that needs the frozen backbone after it loads."""
    if 'network' not in prepared:
        return prepared
    metadata = prepared['metadata']
    validate_backbone(metadata, config, model)
    network = prepared['network']
    context_data = prepared['context_data']
    if context_data is None:
        if raw is None:
            raise ValueError('Encoding requires source text for backbone conditioning')
        if metadata['format'] == NEMOTRON_FORMAT:
            contract = metadata['training']['contract']
            limit = contract['sequence_length'] - 1
        else:
            limit = metadata['encoder']['max_length']
        ids = tokenizer.encode(raw.decode('utf-8'), add_special_tokens=False,
                               truncation=True, max_length=limit)
        if not ids:
            if tokenizer.eos_token_id is None:
                raise ValueError('Backbone conditioning requires nonempty tokenized text or EOS')
            ids = [tokenizer.eos_token_id]
        if metadata['format'] == NEMOTRON_FORMAT:
            ids = (ids + [tokenizer.eos_token_id])[:contract['max_doc_tokens']]
        if network.config.architecture == 'perceiver':
            device = next(model.parameters()).device
            clean = torch.tensor([ids], device=device)
            if metadata['format'] == NEMOTRON_FORMAT:
                states = capture_pre_layer_states(model, network.spec, clean,
                    labels=clean, masked_indices=torch.zeros_like(clean, dtype=torch.bool),
                    skip_loss=True, use_cache=False)
            else:
                states = capture_pre_layer_states(model, network.spec, clean, use_cache=False)
            with torch.no_grad():
                context_data = embedding_bytes(network.cpu().float().eval().encode_hidden_states(
                    [state.cpu() for state in states]).numpy())
        else:
            with torch.no_grad():
                device = next(model.parameters()).device
                values = model.get_input_embeddings()(torch.tensor(ids, device=device))
                context_data = embedding_bytes(values.float().mean(dim=0).cpu().numpy())
    context = read_embedding(context_data, network.context_dim)
    factors, fingerprint = generate(network, context)
    saved = prepared['saved']
    if saved and fingerprint != saved['adapter_sha256']:
        raise ValueError('Regenerated hypernetwork adapter checksum mismatch')
    header = dict(**prepared['identity'], embedding_dim=network.context_dim,
                  context_sha256=hashlib.sha256(context_data).hexdigest(),
                  adapter_sha256=fingerprint,
                  generator=('cpu_fp32_perceiver_v1' if network.config.architecture == 'perceiver'
                             else 'cpu_fp32_v1'))
    return dict(metadata=metadata, header=header, context=context_data,
                factors=factors, rank=network.config.rank, alpha=network.config.alpha,
                personalization_seconds=time.perf_counter() - prepared['started'])


def attach(prepared, config, model):
    validate_backbone(prepared['metadata'], config, model)
    device = next(model.parameters()).device
    factors = {name: tuple(value.to(device) for value in pair)
               for name, pair in prepared['factors'].items()}
    return generated_lora(model, factors, prepared['alpha'], prepared['rank'])
