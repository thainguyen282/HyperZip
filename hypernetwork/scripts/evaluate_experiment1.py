#!/usr/bin/env python3
"""Run lossless Nemotron + hypernetwork evaluations for experiment 1."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_backbone(path):
    recipe = path / 'model' / 'hypernetwork_config.json'
    config = recipe if recipe.is_file() else path / 'config.json'
    metadata = json.loads(config.read_text(encoding='utf-8'))
    if recipe.is_file():
        contract = metadata['contract']
        if contract.get('context') not in ('frozen_token_embedding_mean_v1',
                                           'frozen_pre_layer_hidden_states_perceiver_v1',
                                           'prepared_document_encoder_v1'):
            raise ValueError('The compression runtime does not support this Nemotron conditioning')
        backbone = {'model': 'nemotron', 'model_path': contract['model'], 'revision': contract['revision']}
        weights = path / 'model' / 'hypernetwork.safetensors'
    else:
        backbone = metadata['backbone']
        weights = path / 'hypernetwork.safetensors'
    if backbone['model'] != 'nemotron' or not weights.is_file():
        raise ValueError('Expected a complete Nemotron hypernetwork checkpoint')
    return backbone, config, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trained hypernetwork checkpoint directory')
    parser.add_argument('--enwiki9', type=Path, required=True, help='UTF-8 enwiki9 file')
    parser.add_argument('--fineweb', type=Path, default=ROOT / 'experiments/inputs/fineweb_validation.txt')
    parser.add_argument('--sql', type=Path, default=ROOT / 'datasets/sql_spider_test.txt')
    parser.add_argument('--dna', type=Path, default=ROOT / 'datasets/gene_dnacorpus_test.txt')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--python', default=sys.executable, help='Python with Nemotron dependencies')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--context-size', type=int, default=8192)
    parser.add_argument('--confidence-threshold', type=float, default=0.1)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.context_size < 1 or not 0 <= args.confidence_threshold <= 1:
        parser.error('batch size and context size must be positive, and threshold must be between 0 and 1')
    checkpoint = args.checkpoint.resolve()
    inputs = {name: path.resolve() for name, path in
              [('fineweb', args.fineweb), ('sql', args.sql), ('dna', args.dna), ('enwiki9', args.enwiki9)]}
    for path in inputs.values():
        if not path.is_file():
            parser.error(f'Input file does not exist: {path}')
    try:
        backbone, config, weights = checkpoint_backbone(checkpoint)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(f'Cannot use checkpoint: {exc}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'checkpoint': str(checkpoint), 'checkpoint_config_sha256': sha256(config),
                'checkpoint_weights_sha256': sha256(weights), 'backbone': backbone,
                'inputs': {name: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                           for name, path in inputs.items()},
                'settings': {'device': args.device, 'dtype': args.dtype, 'batch_size': args.batch_size,
                             'context_size': args.context_size, 'confidence_threshold': args.confidence_threshold},
                'results': {}}
    manifest_path = output / 'manifest.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    save()
    common = [args.python, str(ROOT / 'main.py'), '--trust_remote_code', '--no_progress']
    if args.local_files_only:
        common.append('--local_files_only')
    for name, source in inputs.items():
        archive, restored = output / f'{name}.bin', output / f'{name}.restored.txt'
        encode = common + ['--step', 'encode', '--input', str(source), '--output', str(archive),
                           '--model', 'nemotron', '--model_path', backbone['model_path'],
                           '--hypernetwork_path', str(checkpoint), '--device', args.device,
                           '--dtype', args.dtype, '--batch_size', str(args.batch_size),
                           '--context_size', str(args.context_size), '--confidence_threshold', str(args.confidence_threshold)]
        if backbone.get('revision'):
            encode += ['--revision', backbone['revision']]
        decode = common + ['--step', 'decode', '--input', str(archive), '--output', str(restored),
                           '--hypernetwork_path', str(checkpoint), '--device', args.device]
        result = {'status': 'running', 'encode_command': encode, 'decode_command': decode}
        manifest['results'][name] = result
        save()
        try:
            started = time.monotonic()
            subprocess.run(encode, cwd=ROOT, check=True)
            result['encode_wall_seconds'] = time.monotonic() - started
            started = time.monotonic()
            subprocess.run(decode, cwd=ROOT, check=True)
            result['decode_wall_seconds'] = time.monotonic() - started
            if sha256(restored) != manifest['inputs'][name]['sha256']:
                raise ValueError(f'Round-trip verification failed for {name}')
            result.update(status='verified', archive_bytes=archive.stat().st_size,
                          metrics_file=str(archive) + '.metrics.json', roundtrip_ok=True)
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            result.update(status='failed', error=str(exc), roundtrip_ok=False)
            save()
            raise
        save()
    print(f'All round trips verified: {manifest_path}')


if __name__ == '__main__':
    main()
