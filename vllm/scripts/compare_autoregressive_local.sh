#!/usr/bin/env bash
# Run in an existing GPU allocation using the original vllm_ac patch.
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir=$(mktemp -d "$PWD/vllm/results/ar-comparison-XXXXXX")
mkdir -p "$run_dir/source"
exec > >(tee -a "$run_dir/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$run_dir/exit-code.txt"' EXIT
cp -r vllm/vllm_ac "$run_dir/source/"
cp vllm/benchmarks/benchmark_autoregressive.py vllm/benchmarks/benchmark_hf_chunks.py "$run_dir/source/"
cp vllm/scripts/compare_autoregressive_local.sh "$run_dir/source/"
export PYTHONPATH="$run_dir/source${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=teacher_forced_scoring VLLM_CUSTOM_PATCHES=TeacherForcedScoring VLLM_AC_ENABLE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi > "$run_dir/gpu.txt"
vllm/.venv/bin/python -m pip freeze > "$run_dir/environment.txt"
echo "Results: $run_dir"
vllm/.venv/bin/python "$run_dir/source/benchmark_autoregressive.py" --output "$run_dir" "$@"
