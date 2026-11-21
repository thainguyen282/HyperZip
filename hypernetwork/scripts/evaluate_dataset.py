#!/usr/bin/env python3
"""Evaluate a clean or personalized codec on directories of UTF-8 documents."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_experiment1 import ROOT, checkpoint_backbone, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', action='append', required=True, metavar='NAME=PATH',
                   help='UTF-8 file or directory of .txt documents; repeat for each dataset')
    p.add_argument('--checkpoint', type=Path, help='Omit for clean Nemotron 3B')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--precision', choices=('native', 'nf4'), default='native')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--context-size', type=int, default=256)
    p.add_argument('--confidence-threshold', type=float, default=0.1)
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args()
    if a.batch_size < 1 or a.context_size < 1 or not 0 <= a.confidence_threshold <= 1:
        p.error('Invalid batch size, context size, or confidence threshold')
    datasets = {}
    for item in a.dataset:
        name, sep, source = item.partition('=')
        if not sep or not name or name in datasets or not name.replace('_', '').replace('-', '').isalnum():
            p.error('Dataset must be a unique NAME=PATH with a simple name')
        source = Path(source).resolve()
        files = sorted(source.rglob('*.txt')) if source.is_dir() else [source]
        if not files or any(not x.is_file() for x in files):
            p.error(f'No input documents: {source}')
        datasets[name] = files
    if a.checkpoint:
        checkpoint = a.checkpoint.resolve()
        try:
            backbone, cfg, weights = checkpoint_backbone(checkpoint)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            p.error(str(exc))
        identity = {'path': str(checkpoint), 'config_sha256': sha256(cfg), 'weights_sha256': sha256(weights)}
    else:
        checkpoint = None
        backbone = {'model': 'nemotron', 'model_path': 'nvidia/Nemotron-Labs-Diffusion-3B-Base',
                    'revision': '4bd78816ba988ca8b71b8f432b00b682572b236c'}
        identity = None
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        hardware = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                                  text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        hardware = None
    report = {'backbone': backbone, 'checkpoint': identity, 'hardware': hardware,
              'settings': {'precision': a.precision, 'batch_size': a.batch_size, 'context_size': a.context_size,
                           'confidence_threshold': a.confidence_threshold, 'device': a.device},
              'throughput_definition': 'tokens divided by codec compression_seconds; excludes tokenization and personalization',
              'wall_time_definition': 'subprocess elapsed time including loading and personalization', 'datasets': {}}
    manifest = out / 'manifest.json'
    def save():
        manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    common = [a.python, str(ROOT / 'main.py'), '--trust_remote_code', '--no_progress']
    if a.local_files_only:
        common.append('--local_files_only')
    for name, files in datasets.items():
        entry = {'documents': [], 'status': 'running'}
        report['datasets'][name] = entry
        dest = out / name
        dest.mkdir()
        for index, source in enumerate(files):
            archive = dest / f'{index:06d}.bin'
            restored = dest / f'{index:06d}.restored.txt'
            encode = common + ['--step', 'encode', '--input', str(source), '--output', str(archive),
                '--model', 'nemotron', '--model_path', backbone['model_path'], '--revision', backbone['revision'],
                '--device', a.device, '--dtype', 'bfloat16', '--weight_precision', a.precision,
                '--batch_size', str(a.batch_size), '--context_size', str(a.context_size),
                '--confidence_threshold', str(a.confidence_threshold)]
            decode = common + ['--step', 'decode', '--input', str(archive), '--output', str(restored), '--device', a.device]
            if checkpoint:
                encode += ['--hypernetwork_path', str(checkpoint)]
                decode += ['--hypernetwork_path', str(checkpoint)]
            result = {'input': str(source), 'sha256': sha256(source), 'original_bytes': source.stat().st_size,
                      'encode_command': encode, 'decode_command': decode, 'status': 'running'}
            entry['documents'].append(result)
            save()
            try:
                start = time.monotonic()
                subprocess.run(encode, cwd=ROOT, check=True)
                result['encode_wall_seconds'] = time.monotonic() - start
                result['metrics'] = json.loads(Path(str(archive) + '.metrics.json').read_text())
                start = time.monotonic()
                subprocess.run(decode, cwd=ROOT, check=True)
                result['decode_wall_seconds'] = time.monotonic() - start
                if sha256(restored) != result['sha256']:
                    raise ValueError('Round-trip checksum mismatch')
                result.update(status='verified', archive_bytes=archive.stat().st_size)
            except (subprocess.CalledProcessError, OSError, ValueError) as exc:
                result.update(status='failed', error=str(exc))
                entry['status'] = 'failed'
                save()
                raise
            save()
        total = sum(x['original_bytes'] for x in entry['documents'])
        size = sum(x['archive_bytes'] for x in entry['documents'])
        entry.update(status='verified', original_bytes=total, archive_bytes=size,
                     compression_ratio=size / total if total else None,
                     context_bytes=sum(x['metrics'].get('context_bytes', 0) for x in entry['documents']))
        save()
    print(f'Verified results: {manifest}')


if __name__ == '__main__':
    main()
