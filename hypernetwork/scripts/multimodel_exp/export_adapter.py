#!/usr/bin/env python3
"""Generate one model-specific PEFT adapter from a shared checkpoint and document."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file, save_file
from peft import LoraConfig
from hypernetwork.embedding import DocumentEncoder
from hypernetwork.multimodel import MultiModelHypernetwork


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--model', required=True, choices=('3b', '8b'))
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args()
    meta = json.loads((a.checkpoint / 'config.json').read_text())
    if meta.get('format') != 'hyperzip_multimodel_v1':
        p.error('Expected a shared multi-model checkpoint')
    c = meta['contract']
    network = MultiModelHypernetwork(c['targets'], **c['generator']).cpu().eval()
    network.load_state_dict(load_file(str(a.checkpoint / 'hypernetwork.safetensors')), strict=True)
    enc = c['encoder']
    encoder = DocumentEncoder(enc['model_path'], revision=enc['revision'], max_length=enc['max_length'],
        pooling=enc['pooling'].removesuffix('_l2_v1'), device=a.device,
        dtype='bfloat16' if a.device.startswith('cuda') else 'float32', trust_remote_code=True,
        local_files_only=a.local_files_only)
    if encoder.metadata['embedding_dim'] != enc['embedding_dim']:
        raise ValueError('Encoder width differs from checkpoint')
    with torch.no_grad():
        factors = network(encoder.encode(a.input.read_text(encoding='utf-8')).cpu(), a.model)
    a.output.mkdir(parents=True, exist_ok=False)
    LoraConfig(r=network.rank, lora_alpha=network.alpha, target_modules=network.module_names,
               inference_mode=True, bias='none', base_model_name_or_path=c['models'][a.model][0]).save_pretrained(a.output)
    tensors = {f'base_model.model.{name}.lora_{letter}.weight': tensor.contiguous()
               for name, pair in factors.items() for letter, tensor in zip(('A', 'B'), pair)}
    save_file(tensors, str(a.output / 'adapter_model.safetensors'))
    print(f'Wrote {a.model} adapter to {a.output}')


if __name__ == '__main__':
    main()
