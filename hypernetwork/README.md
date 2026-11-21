# HyperZip hypernetwork

## Selected embedding configuration

The selected configuration for the embedding experiment is **GTE-Qwen2-1.5B-instruct + learned linear projection to a 512-dimensional context vector**. The encoder produces 1,536 native dimensions; the projection is trained with the hypernetwork. Its Nemotron 3B training configuration is [`gte_qwen2_1_5b_linear_512.yaml`](../experiments/embedding_context_3b/gte_qwen2_1_5b_linear_512.yaml), and its row-specific launcher and the other table rows are in [`scripts/embedding_exp`](scripts/embedding_exp/README.md). This is an experiment selection, not a change to every legacy CLI default.

Native Nemotron document-encoder checkpoints can be used by the compression runtime. See the experiment instructions for data preparation, training, evaluation, and hardware requirements.

Document-conditioned LoRA generation and context-vector archives following
[HyperZip, §3 and Appendix C](https://arxiv.org/html/2609.36357v1).
The document encoder defaults to GTE-Qwen2-1.5B-Instruct (1536 dimensions,
32,000-token truncation). A shared MLP takes the document vector plus learned
32-dimensional layer and module embeddings and generates rank-8 `q_proj` and
`v_proj` factors. Backbone and encoder parameters stay frozen.

This is an implementation of the method, **not an official pretrained checkpoint
or a reproduction of the reported scores**. The paper does not fully specify
MLP width, depth, activation, initialization, or LoRA alpha. Our explicit choices
are two 1280-unit SiLU hidden layers, shape-specific output heads shared across
layers, alpha 8, small random A and zero B initialization. Parameter count depends
on the backbone's actual Q/V shapes.

The original training backend supports **Fast-dLLM v2**. A separate minimal
Nemotron/FineWeb trainer is described below. Offline integration tests also use
a tiny causal Llama.

## Environment

Use `.venv/bin/python` with `requirements.txt`; no extra Python dependencies are
needed. Calling the interpreter directly avoids stale activation paths. Model
weights must be downloaded or cached. Use `--local_files_only` for offline runs.

## 1. Prepare training documents

Supply UTF-8 files (one document per file) or JSONL with one `{"text": "..."}`
object per line. Use a separate training corpus when evaluating held-out
compression. Substitute your actual training corpus path below.

```bash
./.venv/bin/python -m hypernetwork prepare \
  --input datasets/train.jsonl \
  --output datasets/hyperzip_prepared \
  --tokenizer_path /mmfs1/project/phan/tqn/speculative-compress/base_checkpoint/Fast_dLLM_v2_1.5B \
  --embedding_model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
  --device cuda --dtype bfloat16 --trust_remote_code
```

This streams documents into `documents.jsonl` (token IDs and FP32 embeddings) and
`metadata.json`. Long documents are truncated for **embedding only**; the complete
backbone token sequence is retained. Embeddings use last-token pooling and L2
normalization, following the [GTE model card](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct),
without a query instruction. `--revision` pins the backbone tokenizer and
`--encoder_revision` pins the encoder. Do not change local model/tokenizer
contents after preparing the corpus.

## 2. Train

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv/bin/python -m hypernetwork train \
  --data datasets/hyperzip_prepared \
  --model_path /mmfs1/project/phan/tqn/speculative-compress/base_checkpoint/Fast_dLLM_v2_1.5B \
  --output checkpoints/hyperzip_fastdllm \
  --steps 3500 --gradient_accumulation 128 \
  --learning_rate 2e-5 --sequence_length 8192 --block_size 256 \
  --rank 8 --alpha 8 --hidden_dim 1280 --depth 2 \
  --device cuda --dtype bfloat16 --trust_remote_code --local_files_only
```

The single-device trainer uses document microbatches of one, gradient accumulation,
AdamW, cosine decay, 10% warmup, gradient clipping, FP32 hypernetwork parameters,
and BF16 backbone execution. Non-reentrant layer checkpointing reduces activation
memory; `--no_gradient_checkpointing` trades memory for speed. Lower sequence
length and accumulation for a small trial.

The objective samples a partially masked block after a clean prefix and applies
cross-entropy only to masked tokens. It respects Fast-dLLM's one-position logit
shift and the known head slot used by this codec. This explicit corruption
objective is our implementation choice, **not** the checkpoint's internal
complementary-noise training routine. Confidence selection is absent from the
loss. The backbone stays in eval mode to avoid a second noise process, while
gradients flow through prefix computation and generated factors into the
hypernetwork. Original backbone weights remain unchanged.

Outputs are `config.json`, `hypernetwork.safetensors`, and `training.json`.
Only the final checkpoint is saved; optimizer-state resume and distributed
training are not implemented. Output directories must not exist. Full-scale
training and the paper's resulting quality have not been reproduced.

## 3. Compress with a trained checkpoint

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv/bin/python main.py --step encode \
  --input datasets/enwiki8_partial.txt \
  --output results/enwiki8_partial_hyperzip.bin \
  --model fast_dllm \
  --model_path /mmfs1/project/phan/tqn/speculative-compress/base_checkpoint/Fast_dLLM_v2_1.5B \
  --hypernetwork_path checkpoints/hyperzip_fastdllm \
  --attention_block_size 256 --small_block_size 8 \
  --confidence_threshold 0.1 \
  --device cuda --dtype bfloat16 --trust_remote_code
```

The document encoder is released before the compression backbone loads. For a
precomputed vector, add `--context_embedding document.npy`: one finite vector
with the trained dimension, generated with the same encoder and pooling.
No encoder model loads in that case. `--hypernetwork_path` and `--lora_path`
are mutually exclusive.

The new `session_compression_hypernetwork_v1` ZIP stores `context.f32` alongside
the payload and metadata: **the exact FP32 vector used for generation**, not an
adapter. Both encoding and decoding regenerate factors through a canonical CPU
FP32 path and verify their fingerprint. Context and checkpoint checksums are
validated, and decoded bytes must match the original hash. Use the same
inference runtime/device for both sides.

The default context costs **6,144 bytes per file**, plus metadata, included in
total compression ratio. Shared backbone/hypernetwork weights remain external
and must be retained. Metrics include `context_bytes` and
`personalization_seconds`. `compression_seconds` still measures compression
alone; it does not establish end-to-end throughput.

## 4. Decode without the original text or encoder

```bash
./.venv/bin/python main.py --step decode \
  --input results/enwiki8_partial_hyperzip.bin \
  --output results/enwiki8_partial_hyperzip.restored.txt \
  --device cuda --trust_remote_code --local_files_only
```

If the checkpoint moved, add `--hypernetwork_path /new/checkpoint/path`; contents
must match. Existing base and saved-LoRA archives retain their original formats.
Incompatible historical runtime-personalization archives remain unsupported.

## Optional: export a PEFT adapter

```bash
./.venv/bin/python -m hypernetwork export \
  --checkpoint checkpoints/hyperzip_fastdllm \
  --context_embedding document.npy \
  --output adapters/document
```

Use the result with `--lora_path adapters/document`. That existing workflow needs
the exported adapter at decode time and excludes its size from archive metrics.