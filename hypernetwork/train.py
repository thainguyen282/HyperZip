"""Hypernetwork training entrypoint; Nemotron delegates to NVIDIA's recipe."""
if __package__ in (None, ""):
    # Support both `python hypernetwork/train.py` and `python -m hypernetwork.train`.
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "hypernetwork"
import json
import math
from contextlib import contextmanager
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from config import ModelConfig, get_diffusion_config
from compression.model_loading import load_model
from .checkpoint import save_checkpoint
from .model import (HyperConfig, build_hypernetwork, capture_pre_layer_states,
                    generated_lora, target_spec)

from .train_config import load_nemotron_config

def main(argv=None):
    from .nemotron import HypernetworkDiffusionLMSFTRecipe
    cfg = load_nemotron_config(argv)
    print(cfg, flush=True)
    recipe = HypernetworkDiffusionLMSFTRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == '__main__':
    main()
