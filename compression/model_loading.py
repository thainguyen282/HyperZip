"""Lazy model loading; no imports from the experiment scripts."""
import importlib.metadata
import os
import random
from pathlib import Path
import shutil
import sys

from config import AUTOREGRESSIVE_MODELS, DIFFUSION_MODELS


def load_model(config, modality="text", *, expected_lora=None):
    """Configure inference and load the checkpoint, ready to tokenize and compress."""
    if modality != "text":
        raise ValueError(f"Unsupported modality: {modality!r}")
    if config.model not in (*AUTOREGRESSIVE_MODELS, *DIFFUSION_MODELS, "omni"):
        raise ValueError(f"Unknown model {config.model!r}; choose autoregressive, fast_dllm, nemotron, or omni")
    # Validate adapter identity before allocating the base model.
    from .lora import inspect_lora, load_lora
    lora = inspect_lora(config.lora_path, expected_lora)
    if lora:
        config.lora_path = lora["path"]
    configure_runtime(config)
    if config.model in AUTOREGRESSIVE_MODELS:
        model = load_autoregressive(config)
    elif config.model == "fast_dllm":
        model = load_fast_dllm(config)
    elif config.model == "nemotron":
        model = load_nemotron(config)
    else:
        model = load_omni(config)
    if lora:
        model.model = load_lora(model.model, lora)
    model.lora_metadata = lora
    return model


def configure_runtime(config):
    """Resolve device/dtype and deterministic inference settings before loading."""
    _configure_cuda_home()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    config.device = str(torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    ))
    if config.dtype == "auto":
        config.dtype = "bfloat16" if config.device.startswith("cuda") else "float32"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if Path(config.model_path).exists():
        config.model_path = str(Path(config.model_path).resolve())
    if config.tokenizer_path and Path(config.tokenizer_path).exists():
        config.tokenizer_path = str(Path(config.tokenizer_path).resolve())
    import numpy as np
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    return torch


def _tokenizer(config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.tokenizer_path or config.model_path,
        revision=config.revision if not config.tokenizer_path else None,
        trust_remote_code=config.trust_remote_code, local_files_only=config.local_files_only,
    )


def load_autoregressive(config):
    """Load the causal-LM architecture selected by the checkpoint configuration."""
    from .model import AutoregressiveModel
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = _tokenizer(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, revision=config.revision,
        trust_remote_code=config.trust_remote_code, local_files_only=config.local_files_only,
        torch_dtype=getattr(torch, config.dtype), attn_implementation="eager",
    ).to(config.device).eval()
    config.revision = getattr(model.config, "_commit_hash", None) or config.revision
    return AutoregressiveModel(model, tokenizer, config.device, settings=config.autoregressive)


def _omni_environment():
    installed = importlib.metadata.version("transformers")
    if installed != "4.51.3":
        raise RuntimeError(f"Omni requires transformers==4.51.3; found {installed}. Use the dedicated Omni environment.")
    # Upstream FunASR invokes pip. Constrain its dependencies in this process.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    constraints = str(Path(__file__).resolve().parents[1] / "requirements-omni.txt")
    existing = os.environ.get("PIP_CONSTRAINT", "").split()
    if constraints not in existing:
        os.environ["PIP_CONSTRAINT"] = " ".join(existing + [constraints])


def _configure_cuda_home():
    if not os.environ.get("CUDA_HOME"):
        nvcc = shutil.which("nvcc")
        candidates = [os.environ.get("CUDA_PATH"), os.environ.get("EBROOTCUDA")]
        if nvcc:
            candidates.append(str(Path(nvcc).resolve().parent.parent))
        candidates.append("/usr/local/cuda")
        for candidate in candidates:
            if candidate and (Path(candidate) / "bin/nvcc").is_file():
                os.environ["CUDA_HOME"] = candidate
                break


def load_omni(config):
    if not config.trust_remote_code:
        raise ValueError("Omni requires --trust_remote_code to load its checkpoint-provided implementation")
    _omni_environment()
    from .model import OmniModel
    import torch
    from transformers import AutoModel

    tokenizer = _tokenizer(config)
    model, info = AutoModel.from_pretrained(
        config.model_path, revision=config.revision, trust_remote_code=True,
        local_files_only=config.local_files_only, torch_dtype=getattr(torch, config.dtype),
        attn_implementation="eager", output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Omni checkpoint did not load exactly: {info}")
    if importlib.metadata.version("transformers") != "4.51.3":
        raise RuntimeError("An upstream installer changed Transformers during model loading")
    model = model.to(config.device).eval()
    config.revision = getattr(model.config, "_commit_hash", None) or config.revision
    return OmniModel(model, tokenizer, config.device)


def load_fast_dllm(config):
    """Load the native Fast-dLLM implementation in its dedicated environment."""
    if not config.trust_remote_code:
        raise ValueError("Fast-dLLM requires --trust_remote_code for checkpoint-provided Python")
    if importlib.metadata.version("transformers") != "4.53.1":
        raise RuntimeError("Fast-dLLM requires transformers==4.53.1; install requirements.txt in a separate environment")
    from .model import FastDLLMModel
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    if not Path(config.model_path).is_dir():
        config.model_path = snapshot_download(
            config.model_path, revision=config.revision, local_files_only=config.local_files_only,
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.bin", "*.txt", "*.jinja"],
        )
    if config.tokenizer_path and not Path(config.tokenizer_path).is_dir():
        config.tokenizer_path = snapshot_download(
            config.tokenizer_path, local_files_only=config.local_files_only,
            allow_patterns=["*.json", "*.py", "*.txt", "*.jinja", "*.model"],
        )
    tokenizer = _tokenizer(config)
    model, info = AutoModelForCausalLM.from_pretrained(
        config.model_path, trust_remote_code=True, local_files_only=config.local_files_only,
        torch_dtype=getattr(torch, config.dtype), output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Fast-dLLM checkpoint did not load exactly: {info}")
    if model.config.model_type != "Fast_dLLM_Qwen":
        raise ValueError("Expected a Fast-dLLM Qwen checkpoint")
    model = model.to(config.device).eval().requires_grad_(False)
    # Native evaluation calls SDPA directly. Select its deterministic math backend.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    config.revision = getattr(model.config, "_commit_hash", None) or config.revision
    return FastDLLMModel(model, tokenizer, config.device, settings=config.diffusion)


def load_nemotron(config):
    """Load Nemotron's native same-position diffusion head, without generation."""
    if not config.trust_remote_code:
        raise ValueError("Nemotron requires --trust_remote_code for checkpoint-provided Python")
    if int(importlib.metadata.version("transformers").split(".")[0]) < 5:
        raise RuntimeError("Nemotron requires Transformers >= 5; use its prepared environment")
    from .model import NemotronModel
    import torch
    from transformers import AutoModel

    tokenizer = _tokenizer(config)
    model, info = AutoModel.from_pretrained(
        config.model_path, revision=config.revision, trust_remote_code=True,
        local_files_only=config.local_files_only, dtype=getattr(torch, config.dtype),
        attn_implementation="sdpa", output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Nemotron checkpoint did not load exactly: {info}")
    model = model.to(config.device).eval().requires_grad_(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    config.revision = getattr(model.config, "_commit_hash", None) or config.revision
    return NemotronModel(model, tokenizer, config.device, settings=config.diffusion)
