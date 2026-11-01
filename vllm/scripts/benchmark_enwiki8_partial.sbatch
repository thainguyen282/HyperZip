#!/usr/bin/env bash
#SBATCH --job-name=vllm-enwiki-bench
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=vllm-enwiki-bench-%j.log
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the HyperZip repository root}"
run_dir="$PWD/vllm/results/enwiki8-benchmark-${SLURM_JOB_ID}"
mkdir -p "$run_dir/source"
exec > >(tee -a "$run_dir/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$run_dir/exit-code.txt"' EXIT
export PYTHONUNBUFFERED=1
export VLLM_CUSTOM_PATCHES=TeacherForcedScoring
export VLLM_PLUGINS=teacher_forced_scoring
export OMP_NUM_THREADS=8
nvidia-smi > "$run_dir/gpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$run_dir/allocation.txt"
vllm/.venv/bin/python -m pip freeze > "$run_dir/environment.txt"
cp -r vllm/vllm_ac "$run_dir/source/"
cp vllm/benchmarks/benchmark_autoregressive.py vllm/benchmarks/benchmark_hf_chunks.py "$run_dir/source/"
cp vllm/scripts/benchmark_enwiki8_partial.sbatch "$run_dir/source/"
cp datasets/enwiki8_partial.txt "$run_dir/input.txt"
export PYTHONPATH="$run_dir/source${PYTHONPATH:+:$PYTHONPATH}"
vllm/.venv/bin/python "$run_dir/source/benchmark_autoregressive.py" \
  --input "$run_dir/input.txt" --output "$run_dir" "$@"
cat "$run_dir/comparison.md"
