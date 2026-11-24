"""Prepared paired Tulu data and native NeMo diffusion collation."""
import json
from pathlib import Path

import torch


def load_documents(data_dir, split):
    from datasets import load_from_disk
    root = Path(data_dir)
    metadata = json.loads((root / 'metadata.json').read_text())
    if metadata.get('format') != 'cross_model_tulu_v1' or split not in ('train', 'validation'):
        raise ValueError('Expected prepared cross-model Tulu data and a valid split')
    dataset = load_from_disk(str(root / split), keep_in_memory=False)
    if len(dataset) != metadata[f'{split}_examples']:
        raise ValueError('Prepared dataset length does not match metadata')
    return dataset


class PairedCollator:
    """Apply NeMo's collator separately to both tokenizations of one document."""
    def __init__(self, collators):
        if set(collators) != {'3b', '8b'}:
            raise ValueError('Expected one native collator for each backbone')
        self.collators = collators

    def __call__(self, samples):
        if len(samples) != 1:
            raise ValueError('Cross-model conditioning requires local batch size 1')
        sample = samples[0]
        embedding = torch.as_tensor(sample['context_embedding'], dtype=torch.float32)
        if embedding.ndim != 1 or not torch.isfinite(embedding).all():
            raise ValueError('Expected one finite precomputed prompt embedding')
        result = {'context_embedding': embedding}
        for model_id in ('3b', '8b'):
            example = {field: list(sample[f'{model_id}_{field}'])
                       for field in ('input_ids', 'loss_mask')}
            example['attention_mask'] = [1] * len(example['input_ids'])
            if len(example['input_ids']) != len(example['loss_mask']):
                raise ValueError('Token and loss mask lengths differ')
            result[model_id] = self.collators[model_id]([example])
        return result
