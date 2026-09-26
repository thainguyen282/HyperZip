"""Load an externally generated PEFT LoRA adapter; never generate or train one."""
import hashlib
import importlib.metadata
import json
from pathlib import Path


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_lora(actual, expected):
    """Adapter locations may change, but contents and the PEFT runtime may not."""
    if actual is None or expected is None:
        return actual is expected
    return all(actual.get(key) == expected.get(key) for key in
               ("config_sha256", "weights_sha256", "peft_version"))


def inspect_lora(path, expected=None):
    """Validate local adapter files and optional archived identity before loading."""
    if path is None:
        if expected is not None:
            raise ValueError("This archive requires its saved LoRA adapter")
        return None
    root = Path(path).resolve()
    config_path = root / "adapter_config.json"
    weights_path = root / "adapter_model.safetensors"
    for file in (config_path, weights_path):
        if not file.is_file():
            raise FileNotFoundError(f"Missing LoRA adapter file: {file}")
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    if settings.get("peft_type") != "LORA":
        raise ValueError("Only saved PEFT LORA adapters are supported")
    try:
        version = importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Loading --lora_path requires PEFT in the model's environment") from error
    metadata = {
        "path": str(root), "config_sha256": _file_sha256(config_path),
        "weights_sha256": _file_sha256(weights_path), "peft_version": version,
    }
    if expected is not None and not same_lora(metadata, expected):
        raise ValueError("LoRA adapter checksum or PEFT runtime mismatch")
    return metadata


def load_lora(model, metadata):
    """Attach exactly the saved adapter weights in inference mode, without merging."""
    import torch
    from peft import PeftConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    root = Path(metadata["path"])
    config = PeftConfig.from_pretrained(str(root), local_files_only=True)
    config.inference_mode = True
    if isinstance(config.target_modules, (set, list, tuple)):
        modules = {name for name, _ in model.named_modules()}
        missing = [target for target in config.target_modules
                   if not any(name == target or name.endswith("." + target) for name in modules)]
        if missing:
            raise ValueError(f"Target modules not found in the base model: {sorted(missing)}")
    adapted = get_peft_model(model, config, autocast_adapter_dtype=True)
    weights = load_file(str(root / "adapter_model.safetensors"), device="cpu")
    expected = get_peft_model_state_dict(adapted, save_embedding_layers=False)
    if weights.keys() != expected.keys():
        missing = sorted(expected.keys() - weights.keys())
        unexpected = sorted(weights.keys() - expected.keys())
        raise ValueError(f"LoRA adapter keys do not match the model: missing={missing}, unexpected={unexpected}")
    for name, tensor in weights.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"LoRA adapter tensor shape mismatch: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"LoRA adapter contains nonfinite weights: {name}")
    set_peft_model_state_dict(adapted, weights, ignore_mismatched_sizes=False)
    # Check again before accepting the model if an external writer changed files.
    inspect_lora(root, metadata)
    return adapted.eval().requires_grad_(False)
