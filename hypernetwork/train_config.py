import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

def add_traing_arguments(parser):
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resume")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true", help="Save offline W&B logs locally")
    parser.add_argument("--run-group", default="nemotron-hypernetwork-fineweb")     
    parser.add_argument("--nemotron", action="store_true", help="Use Nemotron model")
    parser.add_argument("--fastdllm", action="store_true", help="Use Fast-dLLM model")



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Hypernetwork training configuration")
    add_traing_arguments(parser)
    args = parser.parse_args(argv)
    return args


# The Nemotron entrypoint below uses the pinned NeMo YAML as its source of
# defaults. The older argparse function remains for the Fast-dLLM trainer.
DEFAULT_NEMOTRON_CONFIG = Path(__file__).resolve().parents[1] / "attempt/hypernetwork_nemotron.yaml"


def load_nemotron_config(argv=None):
    """Translate architecture shortcuts and pass all other overrides to NeMo."""
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

    parser = argparse.ArgumentParser(description="Train the Nemotron hypernetwork")
    parser.add_argument("-c", "--config", default=str(DEFAULT_NEMOTRON_CONFIG))
    parser.add_argument("--nemotron", action="store_true", help="Compatibility alias; Nemotron is the default")
    parser.add_argument("--fastdllm", action="store_true", help="Unsupported by this NeMo entrypoint")
    parser.add_argument("--architecture", choices=["mlp"], default="mlp",
                        help="Hypernetwork generator architecture currently implemented")
    parser.add_argument("--data", help="Prepared dataset for model, train, and validation")
    parser.add_argument("--output", help="Native checkpoint output directory")
    parser.add_argument("--steps", type=int, help="Number of optimizer updates")
    parser.add_argument("--resume", help="Native checkpoint directory to restore")
    parser.add_argument("--width", type=int, help="MLP hidden width")
    parser.add_argument("--depth", type=int, help="MLP hidden layers")
    parser.add_argument("--rank", type=int, help="Generated LoRA rank")
    parser.add_argument("--alpha", type=float, help="Generated LoRA scale")
    parser.add_argument("--max-doc-tokens", type=int, help="Real conditioning tokens")
    parser.add_argument("--global-batch-size", type=int, help="Documents per update")
    parser.add_argument("--lr", type=float, help="AdamW learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    args, dotted = parser.parse_known_args(argv)
    if args.fastdllm:
        parser.error("--fastdllm is not supported by this NeMo entrypoint")
    for name in ("steps", "width", "depth", "rank", "max_doc_tokens", "global_batch_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.alpha is not None and args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr must be positive")
    overrides = []
    def add(path, value):
        if value is not None:
            overrides.extend((f"--{path}", str(value)))
    if args.data:
        for path in ("model.data_dir", "dataset.data_dir", "validation_dataset.data_dir"):
            add(path, args.data)
    add("checkpoint.checkpoint_dir", args.output)
    add("step_scheduler.max_steps", args.steps)
    add("checkpoint.restore_from", args.resume)
    for name in ("width", "depth", "rank", "alpha", "max_doc_tokens"):
        add(f"model.{name}", getattr(args, name))
    add("step_scheduler.global_batch_size", args.global_batch_size)
    add("optimizer.lr", args.lr)
    add("seed", args.seed)
    return parse_args_and_load_config(argv=["-c", args.config, *overrides, *dotted])
