"""Memory-mapped document datasets for long hypernetwork runs."""
import hashlib
import json
from pathlib import Path

FORMAT = 'nemotron_hypernetwork_arrow_v1'
MODEL = 'nvidia/Nemotron-Labs-Diffusion-8B-Base'
REVISION = '59ff0ffee284112fc6ccf37493ced41a11031434'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest(data_dir, sequence_length, verify_files=False):
    root = Path(data_dir)
    raw = (root / 'metadata.json').read_bytes()
    metadata = json.loads(raw)
    if (metadata.get('format') != FORMAT or metadata.get('model') != MODEL or
            metadata.get('model_revision') != REVISION or metadata.get('sequence_length') != sequence_length):
        raise ValueError('Dataset/model/tokenizer mismatch')
    for split in ('train', 'validation'):
        stats = metadata['splits'][split]
        if stats['documents'] < 1 or stats['tokens'] < 2 * stats['documents']:
            raise ValueError('Invalid dataset counts')
    for name, expected in metadata['files'].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Dataset file missing or changed: {name}')
        if verify_files and file_sha256(path) != expected['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    return metadata, hashlib.sha256(raw).hexdigest()


def load_arrow_documents(data_dir, split, seq_length=1024):
    from datasets import load_from_disk
    metadata, _ = manifest(data_dir, seq_length)
    dataset = load_from_disk(str(Path(data_dir) / split), keep_in_memory=False)
    if len(dataset) != metadata['splits'][split]['documents']:
        raise ValueError('Dataset row count does not match manifest')
    return dataset.select_columns(['input_ids', 'loss_mask', 'attention_mask'])
