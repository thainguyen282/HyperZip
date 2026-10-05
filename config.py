"""CLI and model settings for lossless compression."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

OMNI_REVISION = "2233ee1ae2aa2b63e00466197afdeea0ca9e0901"
QWEN_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
FAST_DLLM_MASK_ID = 151665
MODEL_PATHS = {
    "autoregressive": QWEN_PATH,
    "qwen": QWEN_PATH,
    "omni": "lijiang/Omni-Diffusion",
    "nemotron": "nvidia/Nemotron-Labs-Diffusion-3B",
    "fast_dllm": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
}
AUTOREGRESSIVE_MODELS = ("autoregressive", "qwen")
DIFFUSION_MODELS = ("fast_dllm", "nemotron")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_int(value, name, minimum=1):
    _require(type(value) is int and value >= minimum,
             f"{name} must be an integer >= {minimum}")


def _boolean(value, name):
    _require(type(value) is bool, f"{name} must be boolean")


@dataclass
class AutoregressiveConfig:
    use_kv_cache: bool = True
    context_tokens: int | None = None

    def __post_init__(self):
        _boolean(self.use_kv_cache, "Causal cache setting")
        if self.context_tokens is not None:
            _positive_int(self.context_tokens, "Causal context")


@dataclass
class DiffusionConfig:
    # Keep legacy archive defaults separate from CLI defaults.
    block_size: int = 512
    use_kv_cache: bool = False
    use_dual_cache: bool = False
    small_block_size: int = 8
    cache_context_tokens: int | None = None
    confidence_threshold: float | None = None
    fast_inference: bool = False

    def __post_init__(self):
        for name in ("use_kv_cache", "use_dual_cache", "fast_inference"):
            _boolean(getattr(self, name), name)
        _positive_int(self.block_size, "Diffusion block size")
        _positive_int(self.small_block_size, "Small block size")
        if self.confidence_threshold is not None:
            _require(type(self.confidence_threshold) in (int, float)
                     and 0 <= self.confidence_threshold <= 1,
                     "Confidence threshold must be between 0 and 1")
        if self.use_dual_cache:
            _require(self.use_kv_cache, "DualCache requires KV caching")
            _require(self.block_size % self.small_block_size == 0,
                     "Block size must be divisible by small block size for DualCache")
        if self.cache_context_tokens is not None:
            _positive_int(self.cache_context_tokens, "Cache context", self.block_size)


@dataclass
class ModelConfig:
    model: str = "autoregressive"
    model_path: str = QWEN_PATH
    revision: str | None = None
    tokenizer_path: str | None = None
    lora_path: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    seed: int = 0
    diffusion: DiffusionConfig | None = None
    autoregressive: AutoregressiveConfig | None = None

    def __post_init__(self):
        if self.autoregressive is not None:
            _require(self.model in AUTOREGRESSIVE_MODELS
                     and isinstance(self.autoregressive, AutoregressiveConfig),
                     "Only autoregressive models accept AutoregressiveConfig")
        if self.model in DIFFUSION_MODELS:
            if self.diffusion is None:
                self.diffusion = (get_diffusion_config("nemotron")
                                  if self.model == "nemotron" else DiffusionConfig())
            elif not isinstance(self.diffusion, DiffusionConfig):
                raise TypeError("Expected DiffusionConfig for diffusion inference")
        else:
            _require(self.diffusion is None, "Only Fast-dLLM and Nemotron accept diffusion settings")
        if self.diffusion is not None:
            _require(not self.diffusion.fast_inference or self.model == "nemotron",
                     "Fast inference is currently supported for Nemotron only")
            _require(self.model != "nemotron" or not self.diffusion.use_dual_cache,
                     "Nemotron does not support Fast-dLLM DualCache")

    @classmethod
    def from_dict(cls, values):
        """Restore saved settings, including older archive formats."""
        values = dict(values)
        old_model = values.pop("architecture", None)
        if old_model:
            _require(values.get("model", old_model) == old_model,
                     "Conflicting model names in saved settings")
            values["model"] = old_model
        values.setdefault("model", "qwen")
        old_block_size = values.pop("attention_block_size", None)
        diffusion = values.pop("diffusion", None)
        if values["model"] in DIFFUSION_MODELS:
            settings = dict(diffusion or {})
            if old_block_size is not None:
                _require(settings.get("block_size", old_block_size) == old_block_size,
                         "Conflicting diffusion block sizes in saved settings")
                settings["block_size"] = old_block_size
            values["diffusion"] = (
                get_diffusion_config("nemotron", fast_inference=False)
                if values["model"] == "nemotron" and not settings
                else DiffusionConfig(**settings)
            )
        else:
            _require(diffusion is None, "Saved diffusion settings do not apply to this model")
        if values.get("autoregressive") is not None:
            values["autoregressive"] = AutoregressiveConfig(**values["autoregressive"])
        return cls(**values)


def default_model_path(model):
    return MODEL_PATHS.get(model, QWEN_PATH)


def default_model_revision(model):
    return OMNI_REVISION if model == "omni" else None


def get_diffusion_config(model, attention_block_size=None, use_kv_cache=None,
                         use_dual_cache=None, small_block_size=None, cache_context_tokens=None,
                         confidence_threshold=None, fast_inference=None):
    _require(model == "nemotron" or fast_inference is None,
             "--fast_inference requires --model nemotron")
    if model not in DIFFUSION_MODELS:
        options = (attention_block_size, use_dual_cache, small_block_size,
                   cache_context_tokens, confidence_threshold)
        _require(all(value is None for value in options)
                 and (use_kv_cache is None or model in AUTOREGRESSIVE_MODELS),
                 "Diffusion/cache options require --model fast_dllm or nemotron")
        return None

    nemotron = model == "nemotron"
    if nemotron:
        _require(not use_dual_cache and small_block_size is None,
                 "DualCache and --small_block_size require --model fast_dllm")
    kv = True if use_kv_cache is None else bool(use_kv_cache)
    defaults = {
        "block_size": 32 if nemotron else 512,
        "use_kv_cache": kv,
        "use_dual_cache": False if nemotron else (kv if use_dual_cache is None else bool(use_dual_cache)),
        "small_block_size": 32 if nemotron else 8,
        "cache_context_tokens": 8192 if nemotron else None,
        "confidence_threshold": 0.9,
        "fast_inference": nemotron,
    }
    overrides = {
        "block_size": attention_block_size,
        "small_block_size": small_block_size,
        "cache_context_tokens": cache_context_tokens,
        "confidence_threshold": confidence_threshold,
        "fast_inference": None if fast_inference is None else bool(fast_inference),
    }
    defaults.update({key: value for key, value in overrides.items() if value is not None})
    return DiffusionConfig(**defaults)


def diffus_model_config(args, archive_header=None):
    """Legacy public name retained for existing callers."""
    if args.command == "decode":
        _require(archive_header is not None, "Decoding requires saved model settings")
        return ModelConfig.from_dict(archive_header["model"]).diffusion
    return get_diffusion_config(
        args.model, args.attention_block_size, args.use_kv_cache, args.use_dual_cache,
        args.small_block_size, args.cache_context_tokens, args.confidence_threshold, args.fast_inference,
    )


def add_pipeline_arguments(parser):
    group = parser.add_argument_group("Input, output and coding")
    group.add_argument("--input", "--input_file", dest="input_file", required=True)
    group.add_argument("--output", "--AC_output_dir", dest="output", required=True)
    group.add_argument("--metrics_output", help="Default: <output>.metrics.json")
    group.add_argument("--no_progress", action="store_true")
    group.add_argument("--block_size", "--coding_block_size", dest="block_size", type=int, default=128)
    group.add_argument("--frequency_precision", type=int, default=24)
    group.add_argument("--seed", type=int, default=0)


def add_model_arguments(parser):
    group = parser.add_argument_group("Model")
    group.add_argument("--model", default="autoregressive",
                       help="autoregressive, qwen, fast_dllm, nemotron, or omni")
    for name in ("model_path", "revision", "lora_path", "hypernetwork_path", "context_embedding"):
        group.add_argument(f"--{name}")
    group.add_argument("--tokenizer", dest="tokenizer_path")
    group.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    group.add_argument("--device", default="auto")
    for name in ("trust_remote_code", "local_files_only"):
        group.add_argument(f"--{name}", action="store_true")


def add_diffusion_arguments(parser):
    group = parser.add_argument_group("Diffusion")
    for name in ("fast_inference", "use_dual_cache"):
        group.add_argument(f"--{name}", type=int, choices=(0, 1))
    for name in ("attention_block_size", "small_block_size", "cache_context_tokens"):
        group.add_argument(f"--{name}", type=int)
    group.add_argument("--confidence_threshold", type=float)


def add_autoregressive_arguments(parser):
    group = parser.add_argument_group("Autoregressive and KV cache")
    group.add_argument("--context_tokens", type=int)
    group.add_argument("--use_kv_cache", type=int, choices=(0, 1))


def autoregressive_model_config(args):
    if args.model not in AUTOREGRESSIVE_MODELS:
        _require(args.context_tokens is None, "--context_tokens requires --model autoregressive (or qwen)")
        return None
    return AutoregressiveConfig(
        use_kv_cache=True if args.use_kv_cache is None else bool(args.use_kv_cache),
        context_tokens=args.context_tokens,
    )


def get_model_config(args, archive_header=None):
    if args.command == "encode":
        return args.model_config
    _require(archive_header is not None, "Decoding requires the archive's saved model settings")
    config = ModelConfig.from_dict(archive_header["model"])
    config.trust_remote_code = args.trust_remote_code
    config.local_files_only = args.local_files_only
    if args.device != "auto":
        config.device = args.device
    _require(not (args.model_path or args.tokenizer_path),
             "Base-model/tokenizer relocation is not supported; use the archived paths")
    saved_lora = archive_header.get("lora")
    _require(not args.lora_path or bool(saved_lora),
             "Cannot use a LoRA adapter to decode a base-model archive")
    config.lora_path = (args.lora_path or saved_lora["path"]) if saved_lora else None
    return config


def _validate_encode_args(args, parser):
    try:
        _require(not (args.hypernetwork_path and args.lora_path),
                 "Choose --hypernetwork_path or --lora_path")
        _require(not args.context_embedding or bool(args.hypernetwork_path),
                 "--context_embedding requires --hypernetwork_path")
        _positive_int(args.block_size, "Coding block size")
        _require(1 <= args.frequency_precision <= 30, "Frequency precision must be between 1 and 30")
        _require(0 <= args.seed < 2**32, "Seed must be between 0 and 2**32 - 1")
        names = ("model", "tokenizer_path", "lora_path", "device", "dtype",
                 "trust_remote_code", "local_files_only", "seed")
        args.model_config = ModelConfig(
            **{name: getattr(args, name) for name in names},
            model_path=args.model_path or default_model_path(args.model),
            revision=args.revision or default_model_revision(args.model),
            diffusion=diffus_model_config(args),
            autoregressive=autoregressive_model_config(args),
        )
    except ValueError as error:
        parser.error(str(error))


def _yaml_argv(argv):
    """Translate a YAML configuration into the same arguments as the CLI."""
    if not argv or argv[0] != "--config":
        return argv
    _require(len(argv) == 2, "Use --config with exactly one YAML path")
    settings = yaml.safe_load(Path(argv[1]).read_text(encoding="utf-8"))
    _require(isinstance(settings, dict) and settings.get("step") in ("encode", "decode"),
             "Inference YAML must contain step: encode or decode")
    result = ["--step", settings.pop("step")]
    flags = {"trust_remote_code", "local_files_only", "no_progress"}
    for key, value in settings.items():
        if value is None:
            continue
        if key in flags:
            _boolean(value, key)
            if value:
                result.append(f"--{key}")
        else:
            _require(isinstance(value, (str, int, float)), f"{key} must be a scalar CLI value")
            result.extend((f"--{key}", str(int(value) if isinstance(value, bool) else value)))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Independent lossless text compression")
    parser.add_argument("--step", dest="command", choices=("encode", "decode"), required=True)
    for add_arguments in (add_pipeline_arguments, add_model_arguments,
                          add_diffusion_arguments, add_autoregressive_arguments):
        add_arguments(parser)
    args = parser.parse_args(_yaml_argv(list(sys.argv[1:] if argv is None else argv)))
    if args.command == "encode":
        args.metrics_output = args.metrics_output or f"{args.output}.metrics.json"
        _validate_encode_args(args, parser)
    else:
        args.archive = args.input_file
        if args.context_embedding:
            parser.error("--context_embedding applies only to encode")
    return args
