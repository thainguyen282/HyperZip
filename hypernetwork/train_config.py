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
    