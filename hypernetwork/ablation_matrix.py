"""Create and optionally execute a one-factor Nemotron ablation matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


BASELINE = dict(max_doc_tokens=512, width=64, depth=2, rank=4, alpha=4,
                architecture='mlp', targets=['q_proj', 'v_proj'])


def variants():
    yield 'baseline', BASELINE.copy()
    sweeps = {
        'max_doc_tokens': (64, 128, 256, 1024),
        'width': (128, 256),
        'depth': (1, 3),
        'rank': (2, 8, 16),
        'architecture': ('residual_mlp', 'attention', 'perceiver'),
        'targets': (['q_proj', 'k_proj', 'v_proj', 'o_proj'],),
    }
    for factor, values in sweeps.items():
        for value in values:
            settings = BASELINE.copy()
            settings[factor] = value
            if factor == 'rank':
                settings['alpha'] = value  # Hold alpha/rank constant.
            label = '-'.join(value) if isinstance(value, list) else str(value)
            yield f'{factor}_{label}', settings


def command(config, data, output, steps, seed, settings):
    args = [sys.executable, '-m', 'hypernetwork.train', '-c', str(config),
            '--data', str(data), '--output', str(output), '--steps', str(steps),
            '--seed', str(seed)]
    for name, value in settings.items():
        flag = '--' + name.replace('_', '-')
        args.extend((flag, *map(str, value)) if isinstance(value, list) else (flag, str(value)))
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('attempt/hypernetwork_nemotron.yaml'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--execute', action='store_true', help='Run each training command sequentially')
    args = parser.parse_args()
    if args.steps < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Expected positive steps and distinct seeds')
    if not args.config.is_file() or not args.data.is_dir():
        parser.error('Config and prepared data must exist')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    runs = []
    for name, settings in variants():
        for seed in args.seeds:
            path = args.output / f'{name}_seed{seed}'
            runs.append(dict(variant=name, seed=seed, settings=settings,
                             output=str(path), command=command(args.config, args.data,
                                 path, args.steps, seed, settings)))
    (args.output / 'manifest.json').write_text(json.dumps(runs, indent=2) + '\n')
    print(f'Wrote {len(runs)} runs to {args.output / "manifest.json"}', flush=True)
    if args.execute:
        for run in runs:
            print(f'Running {run["variant"]} seed {run["seed"]}', flush=True)
            subprocess.run(run['command'], check=True)


if __name__ == '__main__':
    main()
