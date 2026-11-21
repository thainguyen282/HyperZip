"""Prepare separate tokenized FineWeb documents in the Nemotron Arrow format."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from .fineweb_data import FORMAT, file_sha256

MODEL = 'nvidia/Nemotron-Labs-Diffusion-3B-Base'
REVISION = '4bd78816ba988ca8b71b8f432b00b682572b236c'
DATASET_REVISION = '9bb295ddab0e05d785b879661af7260fed5140fc'


def prepare(args):
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    from .embedding import DocumentEncoder
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
            trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer needs an EOS token')
        encoder = None
        if args.embedding_model:
            encoder = DocumentEncoder(args.embedding_model, revision=args.encoder_revision,
                max_length=args.encoder_max_length, device=args.device, dtype=args.dtype,
                pooling=args.encoder_pooling, trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only)
        if args.input:
            source = load_dataset('json', data_files=str(args.input), split='train', streaming=True)
        else:
            source = load_dataset(args.dataset, args.dataset_config, revision=args.dataset_revision,
                                  split='train', streaming=True).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        stats = {s: {'tokens': 0, 'documents': 0} for s in ('train', 'validation')}
        limits = {'train': args.train_tokens, 'validation': args.validation_tokens}
        seen = sqlite3.connect(str(temp / 'dedup.sqlite'))
        seen.execute('CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID')
        streams = {s: (temp / f'{s}.jsonl').open('w', encoding='utf-8') for s in stats}
        try:
            for row in source:
                text = row.get('text')
                if not isinstance(text, str) or not text.strip():
                    continue
                digest = hashlib.sha256(text.encode('utf-8')).digest()
                split = 'validation' if int.from_bytes(hashlib.sha256(str(args.seed).encode() + digest).digest()[:8], 'big') % 100 == 0 else 'train'
                remaining = limits[split] - stats[split]['tokens']
                if remaining == 0:
                    if all(limits[s] == stats[s]['tokens'] for s in stats):
                        break
                    continue
                if seen.execute('INSERT OR IGNORE INTO seen VALUES (?)', (digest,)).rowcount == 0:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                       max_length=min(args.sequence_length - 1, remaining - 1))
                if not ids:
                    continue
                ids = ids + [tokenizer.eos_token_id]
                if remaining - len(ids) == 1:
                    if len(ids) <= 2:
                        continue
                    ids.pop(-2)
                padding = args.sequence_length - len(ids)
                record = {'input_ids': ids + [tokenizer.eos_token_id] * padding,
                          'loss_mask': [0] + [1] * (len(ids) - 1) + [0] * padding,
                          'attention_mask': [1] * len(ids) + [0] * padding}
                if encoder is not None:
                    record['context_embedding'] = encoder.encode(text).tolist()
                streams[split].write(json.dumps(record, separators=(',', ':')) + '\n')
                stats[split]['tokens'] += len(ids)
                stats[split]['documents'] += 1
            if any(limits[s] != stats[s]['tokens'] or not stats[s]['documents'] for s in stats):
                raise ValueError(f'Source exhausted before token budgets were reached: {stats}')
        finally:
            for stream in streams.values():
                stream.close()
            seen.close()
        for split in stats:
            source_file = temp / f'{split}.jsonl'
            def rows(path=str(source_file)):
                with open(path, encoding='utf-8') as f:
                    for line in f:
                        yield json.loads(line)
            Dataset.from_generator(rows).save_to_disk(str(temp / split))
            source_file.unlink()
        (temp / 'dedup.sqlite').unlink()
        files = {str(p.relative_to(temp)): {'bytes': p.stat().st_size, 'sha256': file_sha256(p)}
                 for p in sorted(temp.rglob('*')) if p.is_file()}
        metadata = {'format': FORMAT, 'model': args.model, 'model_revision': args.revision,
                    'sequence_length': args.sequence_length, 'eos_token_id': tokenizer.eos_token_id,
                    'dataset': str(args.input) if args.input else args.dataset,
                    'dataset_config': None if args.input else args.dataset_config,
                    'dataset_revision': None if args.input else args.dataset_revision,
                    'seed': args.seed, 'split_policy': 'seeded sha256 modulo 100; 0 validation; exact text deduplication',
                    'token_count_policy': 'real tokens including EOS, excluding padding',
                    'splits': stats, 'files': files}
        if encoder is not None:
            metadata['encoder'] = encoder.metadata
        (temp / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        os.rename(temp, output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, help='Local JSONL with a text field; otherwise stream FineWeb')
    p.add_argument('--output', required=True)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--revision', default=REVISION)
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--dataset-revision', default=DATASET_REVISION)
    p.add_argument('--train-tokens', type=int, default=900000000)
    p.add_argument('--validation-tokens', type=int, default=1000000)
    p.add_argument('--sequence-length', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle-buffer', type=int, default=1000)
    p.add_argument('--embedding-model')
    p.add_argument('--encoder-revision')
    p.add_argument('--encoder-pooling', choices=('mean', 'last_token'), default='last_token')
    p.add_argument('--encoder-max-length', type=int, default=1024)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--local-files-only', action='store_true')
    args = p.parse_args()
    if min(args.train_tokens, args.validation_tokens, args.sequence_length, args.shuffle_buffer, args.encoder_max_length) < 2:
        p.error('Token budgets, sequence length, shuffle buffer and encoder length must be at least 2')
    prepare(args)


if __name__ == '__main__':
    main()
