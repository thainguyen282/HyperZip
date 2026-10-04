"""python -m hypernetwork prepare|train|export"""
import argparse
import json
from pathlib import Path


def prepare(args):
    import torch
    from transformers import AutoTokenizer
    from .embedding import DocumentEncoder
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    model_path = str(Path(args.tokenizer_path).resolve()) if Path(args.tokenizer_path).exists() else args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=args.revision,
        trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = args.dtype if args.dtype != 'auto' else ('bfloat16' if device.startswith('cuda') else 'float32')
    encoder = DocumentEncoder(args.embedding_model, revision=args.encoder_revision,
        max_length=args.encoder_max_length, device=device, dtype=dtype,
        trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
    root.mkdir(parents=True)
    count = 0
    with (root / 'documents.jsonl').open('x', encoding='utf-8') as output:
        for filename in args.input:
            path = Path(filename)
            if path.suffix == '.jsonl':
                def texts():
                    with path.open(encoding='utf-8') as stream:
                        for line in stream:
                            if line.strip():
                                yield json.loads(line)[args.text_field]
                iterator = texts()
            else:
                iterator = iter([path.read_bytes().decode('utf-8')])
            for text in iterator:
                if not isinstance(text, str):
                    raise ValueError('Corpus text must be a string')
                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    continue
                item = dict(input_ids=ids, embedding=encoder.encode(text).tolist())
                output.write(json.dumps(item) + '\n')
                count += 1
                if count % 100 == 0:
                    print(json.dumps({'prepared_documents': count}), flush=True)
    if count == 0:
        raise ValueError('Corpus contains no nonempty documents')
    (root / 'metadata.json').write_text(json.dumps(dict(format='hyperzip_training_data_v1',
        tokenizer_path=model_path, tokenizer_revision=args.revision, encoder=encoder.metadata,
        documents=count), indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'prepared_documents': count, 'output': str(root)}))


def export(args):
    import numpy as np
    from .checkpoint import load_checkpoint, export_adapter, embedding_bytes, read_embedding
    network, metadata, _ = load_checkpoint(args.checkpoint)
    context = read_embedding(embedding_bytes(np.load(args.context_embedding, allow_pickle=False)),
                             network.config.embedding_dim)
    export_adapter(args.output, network, context, metadata['backbone']['model_path'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Embed UTF-8 files or JSONL documents once')
    prep.add_argument('--input', nargs='+', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--text_field', default='text')
    prep.add_argument('--tokenizer_path', required=True)
    prep.add_argument('--revision')
    prep.add_argument('--embedding_model', default='Alibaba-NLP/gte-Qwen2-1.5B-instruct')
    prep.add_argument('--encoder_revision')
    prep.add_argument('--encoder_max_length', type=int, default=32000)
    training = commands.add_parser('train', help='Train a Fast-dLLM hypernetwork on prepared documents')
    training.add_argument('--data', required=True)
    training.add_argument('--model_path', required=True)
    training.add_argument('--output', required=True)
    training.add_argument('--steps', type=int, default=3500)
    training.add_argument('--gradient_accumulation', type=int, default=128)
    training.add_argument('--learning_rate', type=float, default=2e-5)
    training.add_argument('--sequence_length', type=int, default=8192)
    training.add_argument('--block_size', type=int, default=256)
    training.add_argument('--hidden_dim', type=int, default=1280)
    training.add_argument('--depth', type=int, default=2)
    training.add_argument('--rank', type=int, default=8)
    training.add_argument('--alpha', type=float, default=8)
    training.add_argument('--seed', type=int, default=0)
    training.add_argument('--no_gradient_checkpointing', action='store_true')
    for sub in (prep, training):
        sub.add_argument('--device', default='auto')
        sub.add_argument('--dtype', choices=('auto', 'float32', 'bfloat16'), default='auto')
        sub.add_argument('--trust_remote_code', action='store_true')
        sub.add_argument('--local_files_only', action='store_true')
    out = commands.add_parser('export', help='Export generated factors as a saved PEFT adapter')
    out.add_argument('--checkpoint', required=True)
    out.add_argument('--context_embedding', required=True)
    out.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'train':
        from .train import train
        train(args)
    else:
        export(args)


if __name__ == '__main__':
    main()
