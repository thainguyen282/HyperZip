"""Compress or decompress text with optional saved or generated LoRA weights."""
from dataclasses import asdict
from contextlib import ExitStack
from pathlib import Path
import gc
import json
import logging
import time

from tqdm.contrib.logging import logging_redirect_tqdm

from config import get_model_config, parse_args
from compression.inputs import read_text_file
from compression.lora import same_lora
from compression.model_loading import load_model
from compression.pipeline import (
    read_archive, validate_output_paths, verify_original, write_archive, write_decoded_file,
)

logger = logging.getLogger(__name__)


def run(args):
    """Read once, load the configured model once, and encode or decode once."""
    if args.command not in {"encode", "decode"}:
        raise ValueError(f"Unknown command: {args.command!r}")
    validate_output_paths(args)
    model = None
    resources = ExitStack()
    try:
        raw = read_text_file(args.input_file) if args.command == "encode" else None
        archive = read_archive(args.archive) if args.command == "decode" else None
        header = archive.header if archive else None
        config = get_model_config(args, header)
        personalized = None
        if args.hypernetwork_path or (header or {}).get("hypernetwork"):
            from hypernetwork.runtime import prepare, attach
            personalized = prepare(args, config, raw=raw, archive=archive)
        logger.info("Loading %s: %s", config.model, config.model_path)
        model = load_model(config, expected_lora=(header or {}).get("lora"))
        if personalized:
            resources.enter_context(attach(personalized, config, model.model))
        logger.info("Model settings: %s", json.dumps(asdict(config)))
        if args.command == "encode":
            return encode_archive(args, model, raw, personalized=personalized)
        return decode_archive(args, model, archive)
    finally:
        resources.close()
        import torch
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def encode_archive(args, model, raw=None, personalized=None):
    """Tokenize and compress one file, then write its archive and metrics."""
    validate_output_paths(args)
    lora = getattr(model, "lora_metadata", None)
    requested = args.model_config.lora_path
    if bool(requested) != bool(lora) or (lora and str(Path(requested).resolve()) != lora["path"]):
        raise ValueError("Load the configured LoRA adapter before encoding")
    if raw is None:
        raw = read_text_file(args.input_file)
    symbols = model.encode_bytes(raw)
    if model.decode_symbols(symbols) != raw:
        raise ValueError("Symbol conversion is not reversible")
    started = time.perf_counter()
    payload = model.compress(
        symbols, args.block_size, args.frequency_precision,
        progress=not args.no_progress, description="Encoding",
    )
    seconds = time.perf_counter() - started
    return write_archive(args, model.schedule, raw, payload, len(symbols), seconds, lora,
                         hypernetwork=personalized)


def decode_archive(args, model, archive=None):
    """Decompress once, verify original bytes, and publish the recovered text."""
    validate_output_paths(args)
    archive = archive if archive is not None else read_archive(args.archive)
    header, metadata = archive.header, archive.metadata
    if model.schedule != header["schedule"]:
        raise ValueError("Model schedule differs from archive")
    if not same_lora(getattr(model, "lora_metadata", None), header.get("lora")):
        raise ValueError("Loaded LoRA adapter does not match the archive")
    started = time.perf_counter()
    symbols = model.decompress(
        archive.payload, metadata["symbol_count"], header["block_size"], header["frequency_precision"],
        progress=not args.no_progress, description="Decoding",
    )
    raw = model.decode_symbols(symbols)
    verify_original(raw, metadata)
    write_decoded_file(args.output, raw)
    seconds = time.perf_counter() - started
    return {
        "records": 1, "decoded_bytes": len(raw), "verified_decode": True,
        "decode_seconds": seconds,
        "decode_tokens_per_second": metadata["symbol_count"] / seconds if seconds > 0 else None,
        "decode_kb_per_second": len(raw) / 1000 / seconds if seconds > 0 else None,
    }


if __name__ == "__main__":
    argv = None
    with logging_redirect_tqdm():
        results = run(parse_args(argv))
    print(json.dumps(results, indent=2))
