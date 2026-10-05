# HyperZip

Lossless text compression with a language model, an optional saved LoRA adapter,
or a **document-conditioned HyperZip hypernetwork**. Compress one UTF-8 file,
recover it exactly, and measure size and speed.

```text
text → tokenizer → model probabilities → arithmetic coding → compressed file
text ← tokenizer ← model probabilities ← arithmetic decoding ← compressed file
```

A saved adapter is loaded once when requested. Alternatively, train a hypernetwork
and use `--hypernetwork_path` to generate an adapter from a document embedding.
The embedding is stored in the archive so decoding can regenerate the same
adapter. See [hypernetwork/README.md](hypernetwork/README.md) for the architecture,
training commands, compression workflow, and differences from the paper.

## Install

For autoregressive models and Fast-dLLM, use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Choose a CUDA-compatible PyTorch installation when using a GPU. CPU is sufficient
for a small autoregressive model. The core dependencies are pinned to the reference
inference environment. No code or data from `speculative-compress` is required.

Nemotron needs its own Python 3.12 environment with `requirements-nemotron.txt`;
Omni needs a separate environment with `requirements-omni.txt`. Their Transformers
versions differ from Fast-dLLM. Those dependency files cover base-model inference;
optional LoRA use also requires PEFT compatible with that environment. Saved LoRA
loading is exercised by CPU tests in the default environment.

## Encode or decode independently

```bash
python main.py --step encode \
  --input examples/sample.txt --output results/text.bin \
  --metrics_output results/text.metrics.json \
  --model autoregressive --model_path HuggingFaceTB/SmolLM2-135M

python main.py --step decode --input results/text.bin --output results/restored.txt
```

Add encode `--lora_path adapters/my_adapter` to use a saved adapter. Decode uses
the model settings and adapter path recorded in the archive. If the adapter was
moved, pass decode `--lora_path /new/location`; the saved config/weight checksums
and PEFT version must match. An adapted archive cannot be decoded without its
adapter. Retain the original base-model weights and tokenizer as well.

Only UTF-8 text files are accepted, read whole into memory. Whitespace, line
endings, Unicode, and missing final newlines are preserved. The tokenizer must
represent the original bytes exactly. Empty input is supported. Directory,
multi-document, and historical runtime-HyperZip archives are not supported. Outputs are
never overwritten. Use the same runtime and inference environment for both
operations; cross-device numerical equivalence is not guaranteed.

## Metrics

Compression prints and saves the same JSON. `--metrics_output` defaults to
`<output>.metrics.json`. For experiment inference, use one YAML file per run:
`python main.py --config experiments/configs/ood/fineweb_validation__nemotron8b_base.yaml`.
The [experiment guide](experiments/README.md) and [training plan](experiments/TRAINING_PLAN.md)
keep checkpoint training separate from `main.py`.

| Field | Definition |
| --- | --- |
| `compressed_file` | Absolute compressed-file path |
| `original_bytes`, `payload_bytes`, `compressed_bytes` | Input, arithmetic payload, and complete archive sizes |
| `token_count` | Number of model tokens encoded |
| `payload_compression_ratio` | Payload bytes / original bytes |
| `total_compression_ratio` | Complete archive bytes / original bytes |
| `payload_bits_per_byte` | 8 × payload bytes / original bytes |
| `total_bits_per_byte` | 8 × complete archive bytes / original bytes |
| `compression_seconds` | Time inside `model.compress` |
| `tokens_per_second` | Token count / compression seconds |

Lower ratios and bits/byte are better. Throughput excludes model/adapter loading,
tokenization, and file I/O. Total-file size includes ZIP metadata but **excludes
external base-model and adapter weights**. An input-specific adapter therefore
has a separate distribution cost; these metrics do not charge that cost.
For the new hypernetwork format, the context vector is included in total size;
shared hypernetwork weights remain external. Additional metrics report
`context_bytes` and `personalization_seconds`.
Experiment runs also report `prediction_groups` and mean tokens per group.
Diffusion runs report observed `refinement_passes`; Nemotron also reports
mean refinement passes per block. These measurements do not change refinement.
Empty-input ratios are null. Decompression reports verified recovery and its own
timing, including decoding, verification, and text output.

The compressed file uses `ZIP_STORED`: ZIP packages the arithmetic-coded payload
and settings without a second compression algorithm. Its members are
`record_000000.bin`, `records.jsonl`, and `manifest.json`.
Hypernetwork archives also contain `context.f32`.

## Read the code

| File | Responsibility |
| --- | --- |
| `main.py` | Direct read → load → compress/decompress → write workflow |
| `config.py` | CLI and saved model settings |
| `compression/inputs.py` | Whole-file UTF-8 reading |
| `compression/model_loading.py`, `lora.py` | Base model and saved-adapter loading |
| `compression/model.py` | Shared token coding loop and model-family wrappers |
| `compression/probability.py`, `arithmeticcoding.py` | Probabilities → integer frequencies → bits |
| `compression/autoregressive_state.py` | Autoregressive context and KV cache |
| `compression/diffusion_state.py`, `fast_dllm_session.py` | Mask refinement and diffusion cache management |
| `compression/pipeline.py` | Single-file archives, integrity checks, and metrics |
| `hypernetwork/` | Document embeddings, LoRA generation, training, and runtime adaptation |

The state modules retain the tokens/cache needed to reproduce predictions during
decoding. Both encoding and decoding share the same prediction and feedback
logic. There are no batch calibration or multi-document archive modules.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m unittest hypernetwork.tests -v
```

The default CPU suite downloads no checkpoints. It covers arithmetic coding,
model schedules and caches, UTF-8 round trips, corruption, output cleanup,
metrics, and a tiny locally generated Llama/PEFT adapter. Run `main.py --step encode` and `main.py --step decode` directly to check a pretrained checkpoint.

## Attribution

The inference pipeline was adapted from the research implementation in
`speculative-compress`. The arithmetic coder derives from Project Nayuki's MIT
implementation; its notice is preserved in
`compression/ARITHMETIC_CODING_LICENSE.txt`. This repository's code is covered by
`LICENSE`.
