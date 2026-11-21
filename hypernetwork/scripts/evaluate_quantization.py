#!/usr/bin/env python3
"""Evaluate HyperZip on FineWeb, enwiki9 and XLDocBench inputs."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from evaluate_experiment1 import ROOT, checkpoint_backbone, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fineweb', type=Path, required=True)
    p.add_argument('--enwiki9', type=Path, required=True, help='UTF-8 enwiki9 subset')
    p.add_argument('--xldocbench', type=Path, required=True, help='UTF-8 XLDocBench input')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--precision', choices=('bf16', '4bit'), default='4bit')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--context-size', type=int, default=256)
    p.add_argument('--confidence-threshold', type=float, default=0.1)
    p.add_argument('--local-files-only', action='store_true')
    args = p.parse_args()
    if args.batch_size < 1 or args.context_size < 1 or not 0 <= args.confidence_threshold <= 1:
        p.error('batch/context sizes must be positive and threshold must be between 0 and 1')
    files = {name: path.resolve() for name, path in [('fineweb', args.fineweb),
             ('enwiki9', args.enwiki9), ('xldocbench', args.xldocbench)]}
    for path in files.values():
        if not path.is_file():
            p.error(f'Input not found: {path}')
    checkpoint = args.checkpoint.resolve()
    try:
        backbone, config, weights = checkpoint_backbone(checkpoint)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        p.error(f'Cannot use checkpoint: {exc}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        hardware = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        hardware = None
    report = {'precision': {'backbone': 'nf4' if args.precision == '4bit' else 'bfloat16', 'generated_lora': 'symmetric_int4' if args.precision == '4bit' else 'float32'},
              'checkpoint': str(checkpoint), 'config_sha256': sha256(config), 'weights_sha256': sha256(weights),
              'backbone': backbone, 'hardware_nvidia_smi': hardware, 'requested_batch_size': args.batch_size,
              'effective_batch_size': 'per input: see codec metrics', 'context_size': args.context_size,
              'confidence_threshold': args.confidence_threshold,
              'throughput_definition': 'codec tokens_per_second: tokens / compression_seconds; excludes model loading, tokenization, and personalization',
              'encode_wall_definition': 'subprocess wall time including loading, personalization, tokenization, coding and archive writing',
              'peak_memory': None, 'peak_memory_note': 'See peak_cuda_memory_bytes in per-input metrics; PyTorch allocated peak, not total device memory.',
              'inputs': {k: {'path': str(v), 'sha256': sha256(v), 'bytes': v.stat().st_size} for k, v in files.items()},
              'results': {}}
    manifest = output / 'manifest.json'
    def save():
        manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    common = [args.python, str(ROOT / 'main.py'), '--trust_remote_code', '--no_progress']
    if args.local_files_only:
        common += ['--local_files_only']
    for name, source in files.items():
        archive, restored = output / (name + '.bin'), output / (name + '.restored.txt')
        encode = common + ['--step', 'encode', '--input', str(source), '--output', str(archive),
            '--model', 'nemotron', '--model_path', backbone['model_path'], '--hypernetwork_path', str(checkpoint),
            '--device', args.device, '--dtype', 'bfloat16', '--weight_precision', 'nf4' if args.precision == '4bit' else 'native', '--batch_size', str(args.batch_size),
            '--context_size', str(args.context_size), '--confidence_threshold', str(args.confidence_threshold)]
        if backbone.get('revision'):
            encode += ['--revision', backbone['revision']]
        decode = common + ['--step', 'decode', '--input', str(archive), '--output', str(restored),
                           '--hypernetwork_path', str(checkpoint), '--device', args.device]
        result = {'status': 'running', 'encode_command': encode, 'decode_command': decode}
        report['results'][name] = result
        save()
        try:
            start = time.monotonic()
            subprocess.run(encode, cwd=ROOT, check=True)
            result['encode_wall_seconds'] = time.monotonic() - start
            result['metrics'] = json.loads(Path(str(archive) + '.metrics.json').read_text())
            start = time.monotonic()
            subprocess.run(decode, cwd=ROOT, check=True)
            result['decode_wall_seconds'] = time.monotonic() - start
            if sha256(restored) != report['inputs'][name]['sha256']:
                raise ValueError('Decoded file hash differs from input')
            result.update(status='verified', roundtrip_ok=True, archive_bytes=archive.stat().st_size)
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            result.update(status='failed', error=str(exc), roundtrip_ok=False)
            save()
            raise
        save()
    print(f'All round trips verified: {manifest}')


if __name__ == '__main__':
    main()
