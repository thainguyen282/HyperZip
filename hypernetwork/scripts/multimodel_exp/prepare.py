#!/usr/bin/env python3
"""Prepare paired Nemotron SFT examples and frozen prompt embeddings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

TULU = 'allenai/tulu-3-sft-mixture'
TULU_REVISION = 'b14afda60f1bbebe55d5d2fa1e4df5042f97f8be'
MODELS = {
    '3b': ('nvidia/Nemotron-Labs-Diffusion-3B-Base', '4bd78816ba988ca8b71b8f432b00b682572b236c'),
    '8b': ('nvidia/Nemotron-Labs-Diffusion-8B-Base', '59ff0ffee284112fc6ccf37493ced41a11031434'),
}
ENCODER = 'Qwen/Qwen3-Embedding-0.6B'


def prompt_messages(messages):
    """Return conversation preceding its final assistant response, never the response."""
    if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'assistant':
        raise ValueError('Expected a conversation ending with an assistant response')
    prompt = messages[:-1]
    if not prompt or not any(m.get('role') == 'user' for m in prompt):
        raise ValueError('Expected at least one user message before the final response')
    return prompt


def prepare(args):
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from nemo_automodel import NeMoAutoTokenizer
    from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
    from hypernetwork.embedding import DocumentEncoder

    if args.output.exists():
        raise ValueError('Output directory already exists')
    if args.token_budget < 1 or args.seq_length < 2 or args.validation_examples < 1:
        raise ValueError('Token budget, sequence length and validation size must be positive')
    snapshot = snapshot_download(TULU, repo_type='dataset', revision=TULU_REVISION,
                                 allow_patterns=['data/*.parquet'], local_files_only=args.local_files_only)
    source = str(Path(snapshot) / 'data')
    tokenizers = {key: NeMoAutoTokenizer.from_pretrained(name, revision=revision,
                  trust_remote_code=True, local_files_only=args.local_files_only)
                  for key, (name, revision) in MODELS.items()}
    datasets = {key: ChatDataset(source, tokenizer, split='train', shuffle_seed=args.seed,
                seq_length=args.seq_length, truncation=True, unshifted=True, mask_history=True)
                for key, tokenizer in tokenizers.items()}
    encoder = DocumentEncoder(ENCODER, revision=args.encoder_revision, max_length=args.encoder_max_length,
                              device=args.device, dtype='float32', pooling='last_token',
                              local_files_only=args.local_files_only)
    if encoder.metadata['revision'] is None:
        raise ValueError('Encoder revision could not be resolved; specify --encoder-revision')
    train, validation, total = [], [], 0
    seen = set()
    for index in range(len(datasets['3b'])):
        raw = datasets['3b'].dataset[index]
        try:
            prompt = prompt_messages(raw.get('messages'))
        except ValueError:
            continue
        document_id = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if document_id in seen:
            continue
        sample3, sample8 = datasets['3b'][index], datasets['8b'][index]
        size = len(sample3['input_ids'])
        if not size or not any(sample3['loss_mask']) or not any(sample8['loss_mask']):
            continue
        is_validation = len(validation) < args.validation_examples
        if not is_validation and total + size > args.token_budget:
            break
        # Chat-template formatting preserves roles and excludes the final answer.
        text = tokenizers['3b'].apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        vector = encoder.encode(text).tolist()
        seen.add(document_id)
        row = {'document_id': document_id, 'context_embedding': vector}
        for model_id, sample in (('3b', sample3), ('8b', sample8)):
            for field in ('input_ids', 'loss_mask'):
                row[f'{model_id}_{field}'] = list(sample[field])
        if is_validation:
            validation.append(row)
        else:
            train.append(row)
            total += size
    if not train or len(validation) < args.validation_examples:
        raise ValueError('Insufficient valid Tulu examples for train and validation')
    args.output.mkdir(parents=True)
    for name, rows in (('train', train), ('validation', validation)):
        Dataset.from_list(rows).save_to_disk(str(args.output / name))
    manifest = {'format': 'cross_model_tulu_v1', 'dataset': TULU, 'dataset_revision': TULU_REVISION,
                'models': MODELS, 'encoder': encoder.metadata, 'prompt': 'all_messages_before_final_assistant_v1',
                'mask_history': True, 'seed': args.seed, 'sequence_length': args.seq_length,
                'budget_3b_tokens': args.token_budget, 'selected_3b_tokens': total,
                'train_examples': len(train), 'validation_examples': len(validation)}
    (args.output / 'metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--token-budget', type=int, default=5_000_000)
    p.add_argument('--validation-examples', type=int, default=32)
    p.add_argument('--seq-length', type=int, default=1024)
    p.add_argument('--encoder-max-length', type=int, default=1024)
    p.add_argument('--encoder-revision', required=True, help='Pinned Qwen3 encoder commit')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--local-files-only', action='store_true')
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
