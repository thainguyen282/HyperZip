# Embedding architecture experiment

Each shell script corresponds to a row of the supplied table. The clean-model
script invokes the base Nemotron codec; every other script starts a separate
hypernetwork training run using its named YAML. No jobs are submitted by these
scripts. Run in a suitable GPU allocation with a Python environment containing
`requirements-nemotron.txt` and NeMo AutoModel:

```bash
export HYPERZIP_PYTHON=/path/to/nemotron/python
bash hypernetwork/scripts/embedding_exp/gte_qwen2_1_5b_linear_512.sh \
  --output checkpoints/my_qwen_512_run --steps 5
```

Omit `--steps 5` for the configured one-epoch training run. Extra arguments
are forwarded to `hypernetwork.train`; use a separate output for every run.
The YAMLs expect prepared FineWeb token IDs and native encoder vectors at their
configured `data_dir`. All three model/train/validation paths can be overridden
together with `--data`. Variants using the same encoder can share a prepared
corpus. Prepare a corpus with the wrapper, for example:

```bash
$HYPERZIP_PYTHON experiments/embedding_context_3b/prepare_dataset.py \
  experiments/embedding_context_3b/gte_qwen2_1_5b_linear_512.yaml
```

For smaller token budgets or local JSONL, use `python -m hypernetwork.prepare_fineweb --help`.
Do not mix inputs or encoder settings across variants.

The bold row uses `gte_qwen2_1_5b_linear_512.sh`: frozen
`Alibaba-NLP/gte-Qwen2-1.5B-instruct`, 1536 native dimensions, and a trained
1536-to-512 projection. The existing full configuration uses Nemotron 3B,
Q/V rank 4 and alpha 4, MLP width 64 and depth 2.

Table interpretation and implementation choices:

* GTE-Small + linear 192 uses a learned 384-to-192 projection. This differs
  from the older `gte_small_192.yaml`, which truncates instead.
* GTE-Base's selected encoder has 768 native dimensions; its 512 row uses
  the existing fixed first-512-coordinate truncation and L2 normalization.
* GTE-Qwen2-1.5B has 1536 native dimensions. To match the displayed 1534,
  this row uses fixed first-1534-coordinate truncation and L2 normalization;
  the separate existing 1536 YAML uses the full native vector. Confirm the
  table's 1534 if exact paper reproduction matters.

The clean-model row requires codec arguments, for example:

```bash
bash hypernetwork/scripts/embedding_exp/clean_model.sh --step encode \
  --input /path/to/fineweb.txt --output results/clean_fineweb.bin --device cuda
```

Train each hypernetwork before held-out FineWeb/OOD evaluation, with no overlap
with training documents. Use `hypernetwork/scripts/evaluate_dataset.py` to evaluate the resulting checkpoints
and verify lossless decoding. The supplied table values are not local
measurements and these scripts do not reproduce them by themselves.
