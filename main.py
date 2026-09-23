"""Coordinate data, models, compression, decompression, and reporting."""
from contextlib import ExitStack, closing, contextmanager, nullcontext
from dataclasses import asdict
from itertools import chain
from pathlib import Path
import gc
import json
import logging
import time

from tqdm.contrib.logging import logging_redirect_tqdm

from config import get_model_config, get_personalization_config, parse_args
from compression.inputs import iter_records
from compression.model_loading import load_model
from compression.personalization import embedding_to_bytes, load_personalizer
from compression.pipeline import (
    ArchiveWriter, DecodedWriter, iter_archive_records, read_archive_header,
    read_record_embedding, verify_original, verify_payload,
)

logger = logging.getLogger(__name__)


def run(args):
    """Load once, then encode or decode records with optional HyperZip adaptation."""
    if args.command not in {"encode", "decode"}:
        raise ValueError(f"Unknown command: {args.command!r}")
    if Path(args.output).exists():
        raise FileExistsError("Output already exists")

    model = personalizer = None
    try:
        with ExitStack() as stack:
            header = None
            if args.command == "encode":
                records = stack.enter_context(closing(iter_records(args)))
                first = next(records, None)  # Validate input before loading weights.
                records = chain([] if first is None else [first], records)
                modality = args.modality
            else:
                header = read_archive_header(args.archive)
                modality = header["modality"]

            config = get_model_config(args, header)
            logger.info("Loading %s: %s", config.model, config.model_path)
            model = load_model(config, modality)
            logger.info("Model settings: %s", json.dumps(asdict(config)))
            personalizer = load_personalizer(
                get_personalization_config(args, header), model, config,
                saved=(header or {}).get("personalization"), encoding=args.command == "encode",
            )
            if personalizer is not None:
                stack.callback(personalizer.close)
            if args.command == "encode":
                return encode_archive(args, model, records, personalizer)
            return decode_archive(args, model, header, personalizer)
    finally:
        import torch
        model = personalizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def encode_archive(args, model, records=None, personalizer=None):
    """Prepare each record, personalize the model, and compress."""
    if args.modality not in model.supported_modalities:
        raise ValueError(f"Unsupported modality: {args.modality!r}")
    if args.personalization != "none" and personalizer is None:
        raise ValueError("Load the personalizer before encoding")

    with ExitStack() as stack:
        if records is None:
            records = stack.enter_context(closing(iter_records(args)))
        archive = stack.enter_context(ArchiveWriter(
            args, model.schedule, personalizer.metadata if personalizer else None,
        ))
        for record in records:
            symbols = model.encode_bytes(record.payload)
            if model.decode_symbols(symbols) != record.payload:
                raise ValueError("Symbol conversion is not reversible")

            side = embedding = None
            if personalizer is not None and symbols:
                started = time.perf_counter()
                data = embedding_to_bytes(personalizer.embed(record.payload), personalizer.dimension)
                side = archive.write_embedding(data)
                embedding = read_record_embedding(
                    archive.output, {"personalization": side}, personalizer.dimension,
                )
                archive.embedding_seconds += time.perf_counter() - started

            started = time.perf_counter()
            adaptation = personalizer.activate(embedding) if side else nullcontext()
            with adaptation as weights_sha256:
                if side:
                    side["weights_sha256"] = weights_sha256
                    archive.adapter_generation_seconds += time.perf_counter() - started
                started = time.perf_counter()
                payload = model.compress(
                    symbols, args.block_size, args.frequency_precision,
                    progress=not args.no_progress, description=f"Encoding record {record.index}",
                )
                archive.write_record(
                    record, payload, len(symbols), time.perf_counter() - started, side,
                )

        return archive.finish()


@contextmanager
def restore_personalization(archive, metadata, personalizer):
    """Reconstruct this record's adapter without access to original text."""
    side = metadata.get("personalization")
    if side is None:
        if personalizer is not None and metadata["symbol_count"]:
            raise ValueError("Missing embedding for a personalized record")
        yield
    else:
        if personalizer is None:
            raise ValueError("Load HyperZip before decoding a personalized record")
        embedding = read_record_embedding(archive, metadata, personalizer.dimension)
        with personalizer.activate(embedding, expected_sha256=side["weights_sha256"]):
            yield


def decode_archive(args, model, header=None, personalizer=None):
    """Restore adaptation and decode each record into its original bytes."""
    header = header or read_archive_header(args.archive)
    if model.schedule != header["schedule"]:
        raise ValueError("Model schedule differs from archive")
    if bool(header.get("personalization")) != (personalizer is not None):
        raise ValueError("Personalizer does not match the archive")

    started = time.perf_counter()
    with DecodedWriter(args.output, header["layout"]) as output:
        with closing(iter_archive_records(args.archive)) as records:
            for metadata, payload in records:
                with restore_personalization(args.archive, metadata, personalizer):
                    raw = recover_record(
                        payload, metadata, model, header["block_size"], header["frequency_precision"],
                        progress=not args.no_progress, description=f"Decoding record {metadata['index']}",
                    )
                output.write(raw, metadata["symbol_count"])
        return output.finish(time.perf_counter() - started)


def recover_record(payload, metadata, model, block_size=128, precision=24,
                   progress=False, description="Decoding"):
    """Decode and validate one archived record."""
    verify_payload(payload, metadata)
    symbols = model.decompress(
        payload, metadata["symbol_count"], block_size, precision, progress, description,
    )
    raw = model.decode_symbols(symbols)
    verify_original(raw, metadata)
    return raw


def main(argv=None):
    """Parse arguments, run the program, and print its results."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with logging_redirect_tqdm():
        results = run(parse_args(argv))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
