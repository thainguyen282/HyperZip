"""Versioned, checksummed hypernetworks and reproducible adapter generation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from compression.lora import _file_sha256
from .model import HyperConfig, build_hypernetwork, target_spec

FORMAT = 'hyperzip_hypernetwork_v1'
NEMOTRON_FORMAT = 'nemotron_hypernetwork_recipe_v2'


def save_checkpoint(path, network, *, backbone, encoder, training=None):
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    metadata = dict(format=FORMAT, **network.configuration(), backbone=backbone,
                    encoder=encoder, training=training or {})
    save_file({k: v.detach().cpu().contiguous() for k, v in network.state_dict().items()},
              str(root / 'hypernetwork.safetensors'))
    (root / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def inspect_checkpoint(path, expected=None):
    root = Path(path).resolve()
    recipe = (root / 'model/hypernetwork_config.json').is_file()
    config_path = root / ('model/hypernetwork_config.json' if recipe else 'config.json')
    weights_path = root / ('model/hypernetwork.safetensors' if recipe else 'hypernetwork.safetensors')
    identity = dict(path=str(root), config_sha256=_file_sha256(config_path),
                    weights_sha256=_file_sha256(weights_path))
    if expected and any(identity[k] != expected.get(k) for k in ('config_sha256', 'weights_sha256')):
        raise ValueError('Hypernetwork checkpoint checksum mismatch')
    metadata = json.loads(config_path.read_text(encoding='utf-8'))
    if recipe and metadata.get('format') == NEMOTRON_FORMAT:
        contract = metadata['contract']
        if contract.get('context') not in ('frozen_token_embedding_mean_v1',
                                           'frozen_pre_layer_hidden_states_perceiver_v1'):
            raise ValueError('Unsupported Nemotron conditioning')
        metadata = dict(format=NEMOTRON_FORMAT, config=contract['hypernetwork'],
                        targets=contract['targets'],
                        backbone=dict(model='nemotron', model_path=contract['model'],
                                      revision=contract['revision']),
                        encoder=dict(pooling=contract['context'],
                                     max_length=contract['max_doc_tokens']),
                        training=dict(contract=contract))
    elif metadata.get('format') != FORMAT:
        raise ValueError('Unsupported hypernetwork checkpoint format')
    HyperConfig(**metadata['config'])
    return identity, metadata


def load_checkpoint(path, expected=None):
    identity, metadata = inspect_checkpoint(path, expected)
    network = build_hypernetwork(HyperConfig(**metadata['config']), metadata['targets'])
    weights = Path(path) / ('model/hypernetwork.safetensors' if metadata['format'] == NEMOTRON_FORMAT
                            else 'hypernetwork.safetensors')
    network.load_state_dict(load_file(str(weights)), strict=True)
    if any(not torch.isfinite(p).all() for p in network.parameters()):
        raise ValueError('Hypernetwork checkpoint contains nonfinite weights')
    inspect_checkpoint(path, identity)
    return network.eval(), metadata, identity


def embedding_bytes(embedding):
    array = np.asarray(embedding, dtype='<f4')
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError('Expected a finite one-dimensional embedding')
    return array.tobytes()


def read_embedding(data, dimension):
    if len(data) != dimension * 4:
        raise ValueError('Context vector size mismatch')
    array = np.frombuffer(data, dtype='<f4').copy()
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite context vector')
    return torch.from_numpy(array)


def generate(network, context):
    """Canonical CPU FP32 inference used by BOTH encoder and decoder."""
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            factors = network.cpu().float().eval()(context.cpu().float())
        digest = hashlib.sha256()
        for name, pair in sorted(factors.items()):
            digest.update(name.encode())
            for tensor in pair:
                digest.update(tensor.contiguous().numpy().astype('<f4', copy=False).tobytes())
        return factors, digest.hexdigest()
    finally:
        torch.set_num_threads(old_threads)


def validate_backbone(metadata, config, model=None):
    expected = metadata['backbone']
    # Local paths are canonicalized by the common model loader.
    actual_path = str(Path(config.model_path).resolve()) if Path(config.model_path).exists() else config.model_path
    if config.model != expected['model'] or actual_path != expected['model_path']:
        raise ValueError('Hypernetwork was trained for a different backbone')
    if expected.get('revision') is not None and config.revision != expected['revision']:
        raise ValueError('Hypernetwork backbone revision mismatch')
    if model is not None and target_spec(model, tuple(sorted({s['module'] for s in metadata['targets']}))) != metadata['targets']:
        raise ValueError('Hypernetwork target projection layout mismatch')


def export_adapter(path, network, context, backbone_path):
    """Export an ordinary PEFT adapter compatible with existing --lora_path."""
    from peft import LoraConfig
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    factors, _ = generate(network, context)
    config = LoraConfig(r=network.config.rank, lora_alpha=network.config.alpha,
                        target_modules=network.module_names, inference_mode=True,
                        base_model_name_or_path=backbone_path, bias='none')
    config.save_pretrained(root)
    tensors = {f'base_model.model.{name}.lora_{letter}.weight': value.contiguous()
               for name, pair in factors.items() for letter, value in zip(('A', 'B'), pair)}
    save_file(tensors, str(root / 'adapter_model.safetensors'))
