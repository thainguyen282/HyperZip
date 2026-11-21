#!/usr/bin/env python3
"""Stream a fixed FineWeb training sample as document-preserving JSONL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tokens', type=int, default=900_000_000)
    p.add_argument('--max-doc-tokens', type=int, default=1023)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle-buffer', type=int, default=1000)
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args()
    if min(a.tokens, a.max_doc_tokens, a.shuffle_buffer) < 1:
        p.error('Token count, document limit and shuffle buffer must be positive')
    if a.output.exists():
        p.error('Output must not exist')
    from datasets import load_dataset
    from transformers import AutoTokenizer
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained('nvidia/Nemotron-Labs-Diffusion-3B-Base',
        revision='4bd78816ba988ca8b71b8f432b00b682572b236c', trust_remote_code=True,
        local_files_only=a.local_files_only)
    stream = load_dataset('HuggingFaceFW/fineweb', 'sample-10BT',
        revision='9bb295ddab0e05d785b879661af7260fed5140fc', split='train', streaming=True)
    stream = stream.shuffle(seed=a.seed, buffer_size=a.shuffle_buffer)
    fd, tmp = tempfile.mkstemp(prefix=a.output.name + '.', suffix='.partial', dir=a.output.parent)
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID')
    total = docs = 0
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            for row in stream:
                text = row.get('text')
                if not isinstance(text, str) or not text.strip():
                    continue
                digest = hashlib.sha256(text.encode()).digest()
                if int.from_bytes(hashlib.sha256(str(a.seed).encode() + digest).digest()[:8], 'big') % 100 == 0:
                    continue
                if not db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (digest,)).rowcount:
                    continue
                ids = tok.encode(text, add_special_tokens=False, truncation=True, max_length=a.max_doc_tokens)
                if not ids:
                    continue
                out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                total += len(ids) + 1
                docs += 1
                if total >= a.tokens:
                    break
        if total < a.tokens:
            raise ValueError(f'FineWeb exhausted after {total} tokens')
        os.link(tmp, a.output)
    finally:
        db.close()
        os.unlink(tmp)
    print(json.dumps({'documents': docs, '3b_retained_tokens': total, 'requested_tokens': a.tokens,
                      'note': 'stopped at a document boundary; each model tokenizes the original text independently'}))


if __name__ == '__main__':
    main()
