"""Command-line and model configuration for the compression pipeline."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
OMNI_REVISION = "2233ee1ae2aa2b63e00466197afdeea0ca9e0901"
HYPERZIP_PATH = str(ROOT / "HyperZip")

MODEL_PATHS = {
    "qwen": str(ROOT / "base_checkpoint/Qwen2.5-0.5B-Instruct"),
    "omni": "lijiang/Omni-Diffusion",
    "nemotron": "nvidia/Nemotron-Labs-Diffusion-3B",
    "fast_dllm": str(ROOT / "base_checkpoint/Fast_dLLM_v2_1.5B"),
}
# The autoregressive family defaults to Qwen; --model_path selects the checkpoint.
MODEL_PATHS["autoregressive"] = MODEL_PATHS["qwen"]
AUTOREGRESSIVE_MODELS = ("autoregressive", "qwen")
DIFFUSION_MODELS = ("fast_dllm", "nemotron")

HYPERNETWORK_CHECKPOINT = str(
    Path(HYPERZIP_PATH)
    / "train_outputs/fineweb_edu_100k/text_to_lora_1.5b_1/checkpoint.pt"
)

# Kept for modules that import these constants directly.
QWEN_PATH = MODEL_PATHS["qwen"]
OMNI_MODEL = MODEL_PATHS["omni"]
FAST_DLLM_PATH = MODEL_PATHS["fast_dllm"]
FAST_DLLM_MASK_ID = 151665
BASE_MODEL = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
PROFILE = "hyperzip_fast_dllm_hooks_v1"


@dataclass
class AutoregressiveConfig:
    use_kv_cache: bool = True
    context_tokens: int | None = None

    def __post_init__(self):
        if type(self.use_kv_cache) is not bool:
            raise ValueError("Causal cache setting must be boolean")
        if self.context_tokens is not None:
            if type(self.context_tokens) is not int or self.context_tokens < 1:
                raise ValueError("Causal context must be a positive integer")


@dataclass
class DiffusionConfig:
    block_size: int = 512
    # False defaults preserve archives written before cache settings existed.
    use_kv_cache: bool = False
    use_dual_cache: bool = False
    small_block_size: int = 8
    cache_context_tokens: int | None = None
    confidence_threshold: float | None = None  # None restores legacy left-to-right archives.
    fast_inference: bool = False  # Missing archive settings retain the original arithmetic schedule.

    def __post_init__(self):
        if type(self.fast_inference) is not bool:
            raise ValueError("Fast inference setting must be boolean")
        if self.confidence_threshold is not None:
            if (type(self.confidence_threshold) not in (int, float)
                    or not 0 <= self.confidence_threshold <= 1):
                raise ValueError("Confidence threshold must be between 0 and 1")
        if type(self.block_size) is not int or self.block_size < 1:
            raise ValueError("Diffusion attention block size must be a positive integer")
        if type(self.use_kv_cache) is not bool or type(self.use_dual_cache) is not bool:
            raise ValueError("Cache settings must be boolean")
        if self.use_dual_cache and not self.use_kv_cache:
            raise ValueError("DualCache requires KV caching")
        if type(self.small_block_size) is not int or self.small_block_size < 1:
            raise ValueError("Small block size must be a positive integer")
        if self.use_dual_cache and self.block_size % self.small_block_size:
            raise ValueError("Attention block size must be divisible by small block size for DualCache")
        if self.cache_context_tokens is not None:
            if type(self.cache_context_tokens) is not int or self.cache_context_tokens < self.block_size:
                raise ValueError("Cache context must hold at least one attention block")


@dataclass
class ModelConfig:
    model: str = "autoregressive"
    model_path: str = QWEN_PATH
    revision: str | None = None
    tokenizer_path: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    seed: int = 0
    diffusion: DiffusionConfig | None = None
    autoregressive: AutoregressiveConfig | None = None

    def __post_init__(self):
        if self.autoregressive is not None:
            if self.model not in AUTOREGRESSIVE_MODELS or not isinstance(self.autoregressive, AutoregressiveConfig):
                raise ValueError("Only autoregressive models accept AutoregressiveConfig")
        if self.model in DIFFUSION_MODELS:
            if self.diffusion is None:
                self.diffusion = get_diffusion_config("nemotron") if self.model == "nemotron" else DiffusionConfig()
            elif not isinstance(self.diffusion, DiffusionConfig):
                raise TypeError("Expected DiffusionConfig for diffusion inference")
        elif self.diffusion is not None:
            raise ValueError("Only Fast-dLLM and Nemotron accept diffusion settings")
        if self.diffusion is not None and self.diffusion.fast_inference and self.model != "nemotron":
            raise ValueError("Fast inference is currently supported for Nemotron only")
        if self.model == "nemotron" and self.diffusion.use_dual_cache:
            raise ValueError("Nemotron does not support Fast-dLLM DualCache")

    @classmethod
    def from_dict(cls, values):
        """Load current settings or migrate an older archive."""
        values = dict(values)
        old_model = values.pop("architecture", None)
        if old_model:
            if values.get("model", old_model) != old_model:
                raise ValueError("Conflicting model names in saved settings")
            values["model"] = old_model

        # Archives that omitted the family used the old Qwen default.
        values.setdefault("model", "qwen")
        old_block_size = values.pop("attention_block_size", None)
        diffusion = values.pop("diffusion", None)
        if values.get("model", "qwen") in DIFFUSION_MODELS:
            settings = dict(diffusion or {})
            if old_block_size is not None:
                if settings.get("block_size", old_block_size) != old_block_size:
                    raise ValueError("Conflicting diffusion block sizes in saved settings")
                settings["block_size"] = old_block_size
            values["diffusion"] = (get_diffusion_config("nemotron", fast_inference=False)
                                   if values["model"] == "nemotron" and not settings
                                   else DiffusionConfig(**settings))
        elif diffusion is not None:
            raise ValueError("Saved diffusion settings do not apply to this model")
        autoregressive = values.get("autoregressive")
        if autoregressive is not None:
            values["autoregressive"] = AutoregressiveConfig(**autoregressive)
        return cls(**values)


@dataclass
class PersonalizationConfig:
    mode: str = "none"
    hyperzip_path: str = HYPERZIP_PATH
    checkpoint_path: str = HYPERNETWORK_CHECKPOINT


def default_model_path(model):
    return MODEL_PATHS.get(model, QWEN_PATH)


def default_model_revision(model):
    return OMNI_REVISION if model == "omni" else None


def get_diffusion_config(model, attention_block_size=None, use_kv_cache=None,
                         use_dual_cache=None, small_block_size=None, cache_context_tokens=None,
                         confidence_threshold=None, fast_inference=None):
    if model != "nemotron" and fast_inference is not None:
        raise ValueError("--fast_inference requires --model nemotron")
    if model == "nemotron":
        if use_dual_cache or small_block_size is not None:
            raise ValueError("DualCache and --small_block_size require --model fast_dllm")
        return DiffusionConfig(
            block_size=32 if attention_block_size is None else attention_block_size,
            use_kv_cache=True if use_kv_cache is None else bool(use_kv_cache),
            use_dual_cache=False, small_block_size=32,
            cache_context_tokens=8192 if cache_context_tokens is None else cache_context_tokens,
            confidence_threshold=0.9 if confidence_threshold is None else confidence_threshold,
            fast_inference=True if fast_inference is None else bool(fast_inference),
        )
    if model != "fast_dllm":
        if any(value is not None for value in (attention_block_size, use_dual_cache,
                                               small_block_size, cache_context_tokens, confidence_threshold)):
            raise ValueError("Diffusion/cache options require --model fast_dllm or nemotron")
        if use_kv_cache is not None and model not in AUTOREGRESSIVE_MODELS:
            raise ValueError("Diffusion/cache options require --model fast_dllm or nemotron")
        return None
    kv = True if use_kv_cache is None else bool(use_kv_cache)
    dual = kv if use_dual_cache is None else bool(use_dual_cache)
    return DiffusionConfig(
        block_size=DiffusionConfig.block_size if attention_block_size is None else attention_block_size,
        use_kv_cache=kv, use_dual_cache=dual,
        small_block_size=DiffusionConfig.small_block_size if small_block_size is None else small_block_size,
        cache_context_tokens=cache_context_tokens,
        confidence_threshold=0.9 if confidence_threshold is None else confidence_threshold,
    )


def diffus_model_config(args, archive_header=None):
    """Get diffusion settings from the CLI or archive (legacy public name)."""
    if args.command == "decode":
        if archive_header is None:
            raise ValueError("Decoding requires saved model settings")
        return ModelConfig.from_dict(archive_header["model"]).diffusion
    return get_diffusion_config(
        args.model, args.attention_block_size, args.use_kv_cache, args.use_dual_cache,
        args.small_block_size, args.cache_context_tokens, args.confidence_threshold, args.fast_inference,
    )


def get_personalization_config(args, archive_header=None):
    saved = (archive_header or {}).get("personalization")
    if args.command == "decode":
        config = PersonalizationConfig(**saved["config"]) if saved else PersonalizationConfig()
    else:
        if args.personalization == "hyperzip" and args.model != "fast_dllm":
            raise ValueError("HyperZip personalization requires --model fast_dllm")
        config = PersonalizationConfig(mode=args.personalization)

    config.hyperzip_path = args.hyperzip_path or config.hyperzip_path
    config.checkpoint_path = args.hypernetwork_checkpoint or config.checkpoint_path
    return config


def get_hypernetwork_args(model_name):
    """Settings expected by HyperZip's make_hypermod function."""
    return SimpleNamespace(
        training_task="sft", target_modules=["q_proj", "v_proj"],
        train_ds_names=["fineweb_edu"], model_dir=model_name,
        use_one_hot_task_emb=False, shared_AB_head=False, autoreg_gen=False,
        learnable_pos_emb=False, learnable_AB_offset=False,
        hypernet_latent_size=512, head_in_size=2048, encoder_type="linear",
        mt_lora_path=None, pred_z_score=True, factorized=False,
        delta_w_scaling=10000,
    )


def add_pipeline_arguments(encode, decode):
    data = encode.add_argument_group("Input")
    source = data.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", "--input_file", dest="input_file")
    source.add_argument("--benchmark_manifest")
    data.add_argument("--collection")
    data.add_argument("--input_format", choices=("text", "jsonl"), default="text")
    data.add_argument("--modality", default="text")
    data.add_argument("--input_chunk_bytes", type=int, default=1024**2)
    data.add_argument("--start_document", type=int, default=0)
    data.add_argument("--max_documents", type=int)
    decode.add_argument("--input", dest="archive", required=True)

    coding = encode.add_argument_group("Arithmetic coding")
    coding.add_argument("--block_size", "--coding_block_size", dest="block_size",
                        type=int, default=128)
    coding.add_argument("--frequency_precision", type=int, default=24)
    encode.add_argument("--seed", type=int, default=0)

    for command in (encode, decode):
        command.add_argument("--output", "--AC_output_dir", dest="output", required=True)
        command.add_argument("--no_progress", action="store_true")


def add_model_arguments(encode, decode):
    model = encode.add_argument_group("Model")
    model.add_argument("--model", default="autoregressive",
                       help="Model family: autoregressive, fast_dllm, nemotron, or omni (qwen is an alias)")
    model.add_argument("--model_path")
    model.add_argument("--revision")
    model.add_argument("--tokenizer", dest="tokenizer_path")
    model.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")

    decode.add_argument("--model_path")
    decode.add_argument("--tokenizer", dest="tokenizer_path")
    for command in (encode, decode):
        command.add_argument("--device", default="auto")
        command.add_argument("--trust_remote_code", action="store_true")
        command.add_argument("--local_files_only", action="store_true")


def add_diffusion_arguments(parser):
    group = parser.add_argument_group("Diffusion and KV cache")
    group.add_argument("--fast_inference", type=int, choices=(0, 1),
                       help="Nemotron fused attention and GPU confidence selection (default: 1)")
    group.add_argument("--confidence_threshold", type=float, help="Refinement confidence (default: 0.9)")
    group.add_argument("--attention_block_size", type=int, help="Diffusion block size (Fast-dLLM: 512; Nemotron: 32)")
    group.add_argument("--use_dual_cache", type=int, choices=(0, 1),
                       help="Fast-dLLM intra-block KV reuse (default: 1 when KV caching is enabled)")
    group.add_argument("--small_block_size", type=int, help="DualCache update span (default: 8)")
    group.add_argument("--cache_context_tokens", type=int,
                       help="Bound KV context (Nemotron: 8192; Fast-dLLM: checkpoint limit)")


def add_autoregressive_arguments(parser):
    group = parser.add_argument_group("Autoregressive inference")
    group.add_argument("--context_tokens", type=int, help="Causal context window (default: checkpoint limit)")
    parser.add_argument("--use_kv_cache", type=int, choices=(0, 1),
                        help="Enable causal or finalized diffusion KV caching (default: 1)")


def autoregressive_model_config(args):
    if args.model not in AUTOREGRESSIVE_MODELS:
        if args.context_tokens is not None:
            raise ValueError("--context_tokens requires --model autoregressive (or qwen)")
        return None
    return AutoregressiveConfig(
        use_kv_cache=True if args.use_kv_cache is None else bool(args.use_kv_cache),
        context_tokens=args.context_tokens,
    )


def add_hyperzip_arguments(encode, decode):
    for command in (encode, decode):
        if command is encode:
            command.add_argument("--personalization", choices=("none", "hyperzip"), default="none")
        command.add_argument("--hyperzip_path")
        command.add_argument("--hypernetwork_checkpoint")


def get_model_config(args, archive_header=None):
    if args.command == "encode":
        return args.model_config
    if archive_header is None:
        raise ValueError("Decoding requires the archive's saved model settings")

    config = ModelConfig.from_dict(archive_header["model"])
    config.trust_remote_code = args.trust_remote_code
    config.local_files_only = args.local_files_only
    if args.device != "auto":
        config.device = args.device
    if args.model_path or args.tokenizer_path:
        if not archive_header.get("personalization"):
            raise ValueError("Model relocation requires a fingerprinted personalized archive")
        config.model_path = args.model_path or config.model_path
        config.tokenizer_path = args.tokenizer_path or config.tokenizer_path
    return config


def _validate_encode_args(args, parser):
    if args.modality != "text":
        parser.error("Only text input is implemented")
    if min(args.input_chunk_bytes, args.block_size) < 1:
        parser.error("Chunk and coding block sizes must be positive")
    if not 1 <= args.frequency_precision <= 30:
        parser.error("Frequency precision must be between 1 and 30")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be between 0 and 2**32 - 1")
    if args.start_document < 0 or (args.max_documents is not None and args.max_documents < 0):
        parser.error("Document offsets and limits must be nonnegative")

    try:
        diffusion = diffus_model_config(args)
        autoregressive = autoregressive_model_config(args)
        get_personalization_config(args)
    except ValueError as error:
        parser.error(str(error))

    args.model_config = ModelConfig(
        model=args.model,
        model_path=args.model_path or default_model_path(args.model),
        revision=args.revision or default_model_revision(args.model),
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        seed=args.seed,
        diffusion=diffusion,
        autoregressive=autoregressive,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Independent lossless text compression")
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode", help="Compress input")
    decode = commands.add_parser("decode", help="Recover an archive")

    add_pipeline_arguments(encode, decode)
    add_model_arguments(encode, decode)
    add_diffusion_arguments(encode)
    add_autoregressive_arguments(encode)
    add_hyperzip_arguments(encode, decode)

    args = parser.parse_args(argv)
    if args.command == "encode":
        _validate_encode_args(args, parser)
    return args
