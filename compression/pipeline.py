"""Single-file archive storage, integrity checks, and compression metrics."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZIP_STORED, ZipFile

FORMAT = "session_compression_v1"
LORA_FORMAT = "session_compression_lora_v1"
HYPERNETWORK_FORMAT = "session_compression_hypernetwork_v1"
CONTEXT_FILE = "context.f32"
QUANTIZATION = "positive_floor_v1"
PAYLOAD_FILE = "record_000000.bin"


@dataclass(frozen=True)
class CompressedFile:
    header: dict
    metadata: dict
    payload: bytes
    context: bytes | None = None


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _validate_destination(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}")
    parent = path.parent
    while not parent.exists() and not parent.is_symlink():
        parent = parent.parent
    if not parent.is_dir():
        raise NotADirectoryError(f"Output parent is not a directory: {parent}")


def validate_output_paths(args):
    """Reject collisions and unusable output parents before loading weights."""
    paths = [Path(args.output)]
    if args.command == "encode":
        paths.append(Path(args.metrics_output))
        source = Path(args.input_file).resolve()
    else:
        paths.append(Path(str(args.output) + ".partial"))
        source = Path(args.archive).resolve()
    resolved = [path.resolve() for path in paths]
    if (resolved[0] == resolved[1] or resolved[0] in resolved[1].parents
            or resolved[1] in resolved[0].parents or source in resolved):
        raise ValueError("Input and output paths must be distinct and not nested")
    for path in paths:
        _validate_destination(path)


def _runtime_versions():
    versions = {}
    for package in ("numpy", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def verify_original(raw, metadata):
    if len(raw) != metadata["original_bytes"] or _sha(raw) != metadata["sha256"]:
        raise ValueError("Decoded bytes failed integrity verification")


def read_archive(path):
    """Read and validate the complete single-file archive before loading weights."""
    if Path(path).is_dir():
        raise ValueError("Directory archives are no longer supported; expected a single compressed file")
    with ZipFile(path) as archive:
        header = json.loads(archive.read("manifest.json"))
        if header.get("format") not in {FORMAT, LORA_FORMAT, HYPERNETWORK_FORMAT} or header.get("personalization") is not None:
            raise ValueError("Unsupported archive format; runtime HyperZip archives are no longer supported")
        if (header.get("quantization") != QUANTIZATION or header.get("modality") != "text"
                or header.get("layout") != "concatenated" or type(header.get("record_count")) is not int
                or header["record_count"] != 1):
            raise ValueError("Expected a single text record with concatenated layout")
        for key, maximum in (("block_size", None), ("frequency_precision", 30)):
            value = header.get(key)
            if type(value) is not int or value < 1 or (maximum and value > maximum):
                raise ValueError(f"Invalid archive setting: {key}")
        lora = header.get("lora")
        if header["format"] == LORA_FORMAT:
            if (not isinstance(lora, dict) or any(not isinstance(lora.get(key), str) or not lora[key]
                    for key in ("path", "config_sha256", "weights_sha256", "peft_version"))):
                raise ValueError("Missing or invalid saved LoRA metadata")
        elif lora is not None or header["model"].get("lora_path") is not None:
            raise ValueError("LoRA metadata requires the saved-LoRA archive format")
        hypernetwork = header.get("hypernetwork")
        context = None
        if header['format'] == HYPERNETWORK_FORMAT:
            if (not isinstance(hypernetwork, dict)
                    or any(not isinstance(hypernetwork.get(key), str) or not hypernetwork[key]
                           for key in ('path', 'config_sha256', 'weights_sha256', 'context_sha256', 'adapter_sha256'))
                    or type(hypernetwork.get('embedding_dim')) is not int
                    or not 0 < hypernetwork['embedding_dim'] <= 65536
                    or hypernetwork.get('generator') != 'cpu_fp32_v1'):
                raise ValueError('Invalid hypernetwork metadata')
            if archive.getinfo(CONTEXT_FILE).file_size != 4 * hypernetwork['embedding_dim']:
                raise ValueError('Context vector size mismatch')
            context = archive.read(CONTEXT_FILE)
            if _sha(context) != hypernetwork['context_sha256']:
                raise ValueError('Context vector checksum mismatch')
        elif hypernetwork is not None:
            raise ValueError('Hypernetwork metadata requires the hypernetwork archive format')
        records = archive.read("records.jsonl")
        if _sha(records) != header["records_sha256"]:
            raise ValueError("Record metadata checksum mismatch")
        lines = records.splitlines()
        if len(lines) != 1:
            raise ValueError("Expected exactly one archived record")
        metadata = json.loads(lines[0])
        if metadata.get("personalization") is not None:
            raise ValueError("Runtime HyperZip records are no longer supported")
        if metadata.get("file") != PAYLOAD_FILE:
            raise ValueError("Invalid payload member name; archive paths must not escape the container")
        for key in ("symbol_count", "original_bytes"):
            if type(metadata.get(key)) is not int or metadata[key] < 0:
                raise ValueError(f"Invalid record size: {key}")
        members = archive.infolist()
        expected_members = {"manifest.json", "records.jsonl", PAYLOAD_FILE}
        if hypernetwork:
            expected_members.add(CONTEXT_FILE)
        if (len(members) != len(expected_members) or {info.filename for info in members}
                != expected_members
                or any(info.compress_type != ZIP_STORED for info in members)):
            raise ValueError("Unexpected or duplicate ZIP_STORED archive members")
        payload = archive.read(PAYLOAD_FILE)
        if _sha(payload) != metadata["payload_sha256"]:
            raise ValueError("Compressed payload checksum mismatch")
    if header["runtime"] != _runtime_versions():
        raise ValueError("Decode with the same NumPy, Torch, and Transformers versions as encoding")
    return CompressedFile(header, metadata, payload, context)


@contextmanager
def _temporary(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def write_archive(args, schedule, raw, payload, token_count, seconds, lora=None, hypernetwork=None,
                  coding_stats=None):
    """Publish a completed ZIP and separate metrics; never invoke a model."""
    validate_output_paths(args)
    if lora and hypernetwork:
        raise ValueError('Cannot combine saved and generated adapters')
    output, metrics_output = Path(args.output), Path(args.metrics_output)
    metadata = {
        "index": 0, "source": str(args.input_file), "file": PAYLOAD_FILE,
        "symbol_count": token_count, "original_bytes": len(raw),
        "sha256": _sha(raw), "payload_sha256": _sha(payload),
    }
    records = (json.dumps(metadata) + "\n").encode("utf-8")
    header = {
        "format": LORA_FORMAT if lora else FORMAT, "quantization": QUANTIZATION,
        "schedule": schedule, "modality": "text", "model": asdict(args.model_config),
        "block_size": args.block_size, "frequency_precision": args.frequency_precision,
        "record_count": 1, "records_sha256": _sha(records),
        "layout": "concatenated", "runtime": _runtime_versions(),
    }
    if lora:
        header["lora"] = lora
    if hypernetwork:
        header['format'] = HYPERNETWORK_FORMAT
        header['hypernetwork'] = hypernetwork['header']
    with _temporary(output) as partial, _temporary(metrics_output) as metrics_partial:
        with ZipFile(partial, "w", compression=ZIP_STORED) as archive:
            archive.writestr(PAYLOAD_FILE, payload)
            if hypernetwork:
                archive.writestr(CONTEXT_FILE, hypernetwork['context'])
            archive.writestr("records.jsonl", records)
            archive.writestr("manifest.json", json.dumps(header, indent=2) + "\n")
        original, compressed_bytes = len(raw), partial.stat().st_size
        metrics = {
            "compressed_file": str(output.resolve()), "original_bytes": original,
            "payload_bytes": len(payload), "compressed_bytes": compressed_bytes,
            "token_count": token_count,
            "payload_compression_ratio": len(payload) / original if original else None,
            "total_compression_ratio": compressed_bytes / original if original else None,
            "payload_bits_per_byte": 8 * len(payload) / original if original else None,
            "total_bits_per_byte": 8 * compressed_bytes / original if original else None,
            "compression_seconds": seconds,
            "tokens_per_second": token_count / seconds if seconds > 0 else None,
        }
        metrics.update(coding_stats or {})
        if args.model_config.model == 'nemotron':
            size = args.model_config.diffusion.block_size
            blocks = (token_count + size - 1) // size
            metrics['mean_refinement_passes_per_block'] = (
                metrics.get('refinement_passes', 0) / blocks if blocks else None)
        if hypernetwork:
            metrics['context_bytes'] = len(hypernetwork['context'])
            metrics['personalization_seconds'] = hypernetwork['personalization_seconds']
        metrics_partial.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        # Publish completed files without overwriting a concurrently created path.
        os.link(partial, output)
        try:
            os.link(metrics_partial, metrics_output)
        except BaseException:
            output.unlink()
            raise
    return metrics


def write_decoded_file(output, raw):
    """Publish verified text; preserve a partial file if output writing fails."""
    output = Path(output)
    _validate_destination(output)
    partial = Path(str(output) + ".partial")
    _validate_destination(partial)
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as stream:
        stream.write(raw)
    os.link(partial, output)
    partial.unlink()
