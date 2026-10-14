"""Summarize paired held-out compression results from JSONL measurements.

Each row needs variant, seed, file_id, input_tokens, input_bytes, archive_bytes,
context_bytes, encode_seconds, decode_seconds, and roundtrip_ok. The caller
must record one row per file for every variant/seed and verify document splits.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def size_bin(n):
    if n < 1024:
        return '<1K'
    for boundary in (2048, 4096, 8192, 16384, 32768):
        if n < boundary:
                return f'{boundary // 2048}K-{boundary // 1024}K' if boundary > 2048 else '1K-2K'
    return '32K+'


def summarize(rows, baseline):
    required = ('variant', 'seed', 'file_id', 'input_tokens', 'input_bytes', 'archive_bytes',
                'context_bytes', 'encode_seconds', 'decode_seconds', 'roundtrip_ok')
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if any(key not in row for key in required):
            raise ValueError('Measurement is missing a required field')
        key = (row['variant'], row['seed'], row['file_id'])
        if key in seen:
            raise ValueError(f'Duplicate measurement: {key}')
        seen.add(key)
        if (row['input_tokens'] <= 0 or row['input_bytes'] <= 0 or row['archive_bytes'] < 0 or row['context_bytes'] < 0
                or row['context_bytes'] > row['archive_bytes'] or
                row['encode_seconds'] < 0 or row['decode_seconds'] < 0 or
                not row['roundtrip_ok']):
            raise ValueError(f'Invalid or failed round trip: {key}')
        groups[(row['variant'], row['seed'])].append(row)
    if not groups or baseline not in {variant for variant, _ in groups}:
        raise ValueError('Baseline or measurements are missing')
    baseline_files = {seed: {r['file_id']: (r['input_tokens'], r['input_bytes']) for r in values}
                      for (variant, seed), values in groups.items() if variant == baseline}
    expected_seeds = set(baseline_files)
    output = []
    for (variant, seed), values in sorted(groups.items()):
        files = {r['file_id']: (r['input_tokens'], r['input_bytes']) for r in values}
        if set(baseline_files) != {s for v, s in groups if v == variant} or seed not in expected_seeds:
            raise ValueError(f'Seed mismatch for {variant}')
        if files != baseline_files[seed]:
            raise ValueError(f'Held-out files differ for {variant}, seed {seed}')
        for bucket, subset in [('all', values), *[(name, [r for r in values if size_bin(r['input_tokens']) == name])
                                               for name in ('<1K', '1K-2K', '2K-4K', '4K-8K',
                                                            '8K-16K', '16K-32K', '32K+')]]:
            if not subset:
                continue
            output.append(dict(variant=variant, seed=seed, size_bin=bucket,
                files=len(subset), input_bytes=sum(r['input_bytes'] for r in subset),
                archive_bytes=sum(r['archive_bytes'] for r in subset),
                context_bytes=sum(r['context_bytes'] for r in subset),
                bits_per_byte=8 * sum(r['archive_bytes'] for r in subset) /
                              sum(r['input_bytes'] for r in subset),
                median_file_bits_per_byte=median(8 * r['archive_bytes'] / r['input_bytes'] for r in subset),
                encode_seconds=sum(r['encode_seconds'] for r in subset),
                decode_seconds=sum(r['decode_seconds'] for r in subset)))
    reference = {(item['seed'], item['size_bin']): item for item in output
                 if item['variant'] == baseline}
    for item in output:
        base = reference[(item['seed'], item['size_bin'])]
        item['delta_bits_per_byte'] = item['bits_per_byte'] - base['bits_per_byte']
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    parser.add_argument('--baseline', default='baseline')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.measurements.read_text().splitlines() if line.strip()]
    report = summarize(rows, args.baseline)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
