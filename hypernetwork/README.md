# HyperZip hypernetwork

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
./.venv/bin/python main.py encode \
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
./.venv/bin/python main.py decode \
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

## Tests

```bash
OMP_NUM_THREADS=1 ./.venv/bin/python -m unittest hypernetwork.tests -v
OMP_NUM_THREADS=1 ./.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Offline tests cover trainable conditioning, frozen base weights, target alignment,
exact PEFT export equivalence, corruption rejection, and round trips after
deleting the source and moving the checkpoint. No pretrained downloads are needed.

## Nemotron hypernetwork recipe / FineWeb smoke test

`HypernetworkDiffusionLMSFTRecipe` in `hypernetwork.nemotron` subclasses NVIDIA's
`DiffusionLMSFTRecipe`. NVIDIA owns the training loop, corruption, hybrid loss,
gradient accumulation, AdamW updates, scheduler, validation, metrics, and native
checkpoint lifecycle. The override generates one document's adapters and calls
`super()._forward_backward_step()` with those adapters attached through backward
and non-reentrant activation recomputation. No upstream source is modified.

The composite model keeps the pinned Nemotron 8B backbone frozen in BF16/eval
mode and trains only the FP32 MLP hypernetwork. Defaults are width 64, depth 2,
rank/alpha 4, Q/V targets, and a mean of the first 512 clean token embeddings as
conditioning. Documents remain separate, with a 1,024-token limit. Padding and
the first token are unsupervised; conditioning excludes padding. This uses the
document itself as conditioning, so its validation loss is not unconditioned
language-model perplexity.

`attempt/hypernetwork_nemotron.yaml` defaults to five optimizer updates, global
batch one, microbatch one, AdamW LR 1e-4 / betas (0.9, 0.98) / epsilon 1e-5 /
weight decay 0.1, constant LR without warmup, and clipping at 1.0. The native
objective is AR + 0.3 times diffusion. Only one GPU is supported; use
`--step_scheduler.global_batch_size 4` for four documents per accumulated update,
keeping `local_batch_size: 1`. Native corruption seeding and validation replace
the old custom loop; old and new loss sequences are not expected to match.

The default corpus is the existing `data/hypernetwork_fineweb_1367451` sample:
32 training documents and eight validation documents. The loader checks the
pinned model/tokenizer, token lengths, and train/validation document separation.
The YAML contains the data path in the model, dataset, and validation dataset
sections; override all three together when selecting a different prepared corpus.
The preparation script remains available separately; the smoke job reuses this
sample and does not download another corpus.

Run the production recipe inside a GPU allocation, from the repository root:

```bash
attempt/.automodel/.venv/bin/python -m hypernetwork.nemotron \
  -c attempt/hypernetwork_nemotron.yaml \
  --checkpoint.checkpoint_dir checkpoints/hypernetwork_recipe_my_smoke
```

The former `--data`, `--output`, `--steps`, and `--resume` interface is replaced
by native YAML/dotted overrides. For an explicit resume, use a new output
directory and the actual native checkpoint subdirectory:

```bash
attempt/.automodel/.venv/bin/python -m hypernetwork.nemotron \
  -c attempt/hypernetwork_nemotron.yaml \
  --step_scheduler.max_steps 6 \
  --checkpoint.checkpoint_dir checkpoints/hypernetwork_recipe_my_resume \
  --checkpoint.restore_from checkpoints/hypernetwork_recipe_my_smoke/epoch_0_step_4
```

Checkpoint directories contain `model/hypernetwork.safetensors` and
`model/hypernetwork_config.json`, plus native optimizer, scheduler, dataloader,
and RNG state. The backbone is referenced by its pinned revision, not copied.
The conditioning/model/data contract must match when restoring. Legacy custom
loop checkpoints are not resumable by this recipe, and this checkpoint format
is not yet integrated into the compression runtime.

Submit the complete acceptance workflow:

```bash
mkdir -p results/hypernetwork-nemotron
sbatch attempt/train_hypernetwork.sbatch
```

One A100 80GB allocation (16 CPUs, 120GB RAM, two hours) runs CPU tests, five smoke
steps, resume to six, an independent six-step reference, and two steps with
accumulation four. It stops on failure and does not continue to 50 steps.
The separate `hypernetwork.smoke_nemotron` acceptance harness checks frozen
backbone hashes, optimizer/scheduler/dataloader restoration, finite metrics,
updated hypernetwork weights, and exact adapter regeneration after reload.
Resume/reference loss, gradient norm, and weights use rtol=1e-4 / atol=1e-6.
These checks are outside the production trainer; there is no second training loop.

Logs are under `results/hypernetwork-nemotron/JOB_ID/`, with the batch log at
`results/hypernetwork-nemotron/slurm-JOB_ID.log`. Checkpoints, native
`training.jsonl` / `validation.jsonl`, per-stage acceptance summaries, and final
`verification.json` are under `checkpoints/hypernetwork_recipe_JOB_ID/`.
Per-step times derived from native throughput include any preceding validation
or checkpoint work; cold setup and total train/validation/checkpoint time are
reported separately. GPU memory is also sampled externally once per second.

W&B is enabled in **offline mode**. The Slurm job stores each stage's W&B run
inside its checkpoint output. Direct launches use W&B's default local directory
unless `--wandb.dir` is supplied. Nothing is uploaded automatically.

### Verified recipe smoke workflow

Slurm job **1371058** completed on one A100 80GB on **2026-10-03**, exit code 0,
in **10m13s**. All six CPU regression tests and all four GPU stages passed.

| Stage | Optimizer steps | Accumulation | Final training loss | Peak allocated GiB |
|---|---:|---:|---:|---:|
| smoke | 0 → 5 | 1 | 4.588711 | 18.944 |
| resume | 5 → 6 | 1 | 6.494614 | 18.936 |
| reference | 0 → 6 | 1 | 6.494614 | 18.944 |
| accumulation | 0 → 2 | 4 | 8.646128 | 18.957 |

The resumed and uninterrupted six-step hypernetwork weights were **bitwise
identical** (maximum difference 0.0). Native optimizer, learning-rate scheduler,
and dataloader state restored successfully. Every stage retained the exact
frozen backbone hash, updated the hypernetwork, and regenerated identical
adapters after checkpoint reload. Adam moments remained FP32. The native
weight checkpoint contains about 15 MB of hypernetwork weights, not a duplicate
of the 8B backbone.

### FineWeb 900M-token run

For architecture experiments, launch the same NeMo recipe through
`hypernetwork/train.py`. The Python config adapter in `train_config.py` loads
the YAML and maps convenient flags to NeMo's dotted settings; any remaining
`--section.key value` settings go directly to NeMo. For example, inside an
A100 allocation, a short run on the existing sample with a wider, deeper MLP:

```bash
attempt/.automodel/.venv/bin/python hypernetwork/train.py \
  -c attempt/hypernetwork_nemotron.yaml \
  --data data/hypernetwork_fineweb_1367451 \
  --output checkpoints/hypernetwork_width96_depth3_trial \
  --steps 5 --width 96 --depth 3 --rank 8 --alpha 8
```

The direct file form and `python -m hypernetwork.train` are equivalent. The
current architecture is an MLP; width, depth, rank, and alpha can vary, but a
Perceiver or other generator would require an additional implementation. Use a
new output directory for each architecture. The legacy Fast-dLLM training
function remains in `train.py`, but this entrypoint now starts Nemotron.

The large run uses the same pinned backbone, hypernetwork, native NeMo trainer,
hybrid loss, and one-GPU settings as the verified smoke test. It changes the
dataset to FineWeb `sample-10BT`, runs one epoch over the prepared documents,
and allows at most 72 hours in one Slurm allocation. It is a training attempt;
the smoke test establishes execution and resume correctness, not convergence.

From the repository root, submit data preparation and then a dependent GPU job:

```bash
mkdir -p results/hypernetwork-nemotron
prep_job=$(sbatch --parsable attempt/prepare_hypernetwork_900m.sbatch)
sbatch --dependency=afterok:"$prep_job" --kill-on-invalid-dep=yes \
  attempt/train_hypernetwork_900m.sbatch
```

Preparation streams the pinned FineWeb revision with seed 42 and exact document
deduplication. It writes `data/hypernetwork_fineweb_900m/` only after producing
**900,000,000 real training tokens** and **1,000,000 separate validation tokens**
and verifying the manifest. Each document is a separate 1,024-position example;
the final EOS counts as a real token, while padding does not. The first token
and padding are excluded from supervision. Datasets are Arrow files loaded
through `datasets.load_from_disk`; the manifest records file hashes and token
counts. The preparation script refuses to overwrite an existing dataset.

The GPU job requests one A100 80GB, 16 CPUs, 120GB, and 72 hours. It uses
`attempt/hypernetwork_fineweb_900m.yaml`: batch one, AdamW LR 1e-4, constant
schedule, validation every 10,000 steps, checkpoint every 500 steps, and a
preemption checkpoint request five minutes before the time limit. It does not
cap training at 50 steps. An epoch contains one update per training document;
the 72-hour limit can end before that epoch completes. The smoke test's roughly
0.64 seconds per update already implies more than 72 hours even if every
document were 1,024 real tokens. Actual document lengths and validation and
checkpoint costs may increase the time. A successful allocation with a partial
checkpoint is therefore expected; it must not be described as 900M tokens
trained unless `run-status.json` reports `completed_one_pass`.

For the submitted attempt, preparation is Slurm job **1371434** and the
dependent GPU run is **1371435**. Check live state and preparation progress with:

```bash
squeue -j 1371434,1371435 -o '%i %T %M %R'
tail -n 5 results/hypernetwork-nemotron/prepare-900m-1371434.log
tail -n 20 results/hypernetwork-nemotron/train-900m-1371435.log
```

The GPU job writes native checkpoints, `training.jsonl`, `validation.jsonl`,
and `run-status.json` under `checkpoints/hypernetwork_900m_1371435/`. Its
allocation record, copied config, detailed train log, and 30-second GPU samples
are under `results/hypernetwork-nemotron/900m_1371435/`; the Slurm batch log is
`results/hypernetwork-nemotron/train-900m-1371435.log`. W&B is enabled in
offline mode and its local files are kept with the checkpoint. Data, results,
and checkpoints are excluded from Git.

If a later allocation is explicitly requested, resume from the actual latest
native checkpoint directory and a fresh output directory. For example, after
replacing the placeholder with the checkpoint printed in `run-status.json`:

```bash
sbatch --export=ALL,RESUME_FROM=/absolute/path/to/epoch_N_step_M \
  attempt/train_hypernetwork_900m.sbatch
```

Resume restores optimizer, scheduler, dataloader, and RNG state as verified in
the smoke test. No follow-on allocation is submitted automatically.

The five-step smoke median after excluding the first two updates was
**0.639 seconds/update**. Initialization,
first-step compilation, validation, full-backbone verification hashes, and
checkpoint work account for most of this short job's elapsed time. Sampled
peak device memory was about 22.3 GiB, distinct from the approximately 18.96 GiB
PyTorch allocated peak. Losses were finite but variable (the smoke run included
a loss of 27.42); these results establish execution and resume correctness,
not convergence or improved compression quality.

Final verification: `checkpoints/hypernetwork_recipe_1371058/verification.json`.
Per-stage reports and checkpoints are below that directory; logs are in
`results/hypernetwork-nemotron/1371058/`, with the full batch log at
`results/hypernetwork-nemotron/slurm-1371058.log`. All four W&B runs were saved
offline under their stage directories.

The initial attempt, job **1371049**, passed training but stopped at the resume
verifier. TorchData keeps restored sampler progress in a pending lazy state;
materializing its iterator initially reports an internal zero counter despite
returning the correct next document. The verifier now checks pending state
without advancing it, and a regression test verifies the next resumed document.
The successful workflow above reran every stage with that fix.

### Historical custom-loop run (before the recipe refactor)

Slurm coordinator job **1367451** completed successfully (exit 0) in 4m16s,
using an idle A100 in allocation 1365490. The first launcher attempt (1367450)
failed before training due to a non-executable `env` on PATH; the launcher now
uses `/usr/bin/env` explicitly.

The prepared sample contains 32 training documents (16,881 tokens) and 8 held-out
documents (3,347 tokens). Five smoke updates plus resumed updates 6–50 processed
26,970 document tokens, including repeated visits. All 50 updates had finite,
nonzero AR/diffusion losses and hypernetwork gradients. The 3,732,736-parameter
hypernetwork changed; the full frozen backbone hash remained identical. Optimizer
resume and exact generated-adapter equality after reload both passed.

Median warm update time was 0.646s and peak allocated GPU memory was 17.71 GiB.
Fixed-mask validation loss was 4.870802 before training, 4.904582 at step five,
and 4.868983 at step 50. This tiny change establishes successful execution, not
a meaningful quality improvement or reproduction of paper results.

The final checkpoint and summary are in
`checkpoints/hypernetwork_nemotron_1367451/train/`; the complete batch log is
`results/hypernetwork-nemotron/slurm-1367451.log`. W&B logs were saved offline.
