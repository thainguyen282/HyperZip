#!/usr/bin/env bash
#SBATCH --job-name=vllm-codec-bench
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=vllm-codec-bench-%j.log
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the HyperZip repository root}"
run_dir="$PWD/vllm/results/benchmark-${SLURM_JOB_ID}"
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$run_dir/exit-code.txt"' EXIT
export PYTHONUNBUFFERED=1
export VLLM_CUSTOM_PATCHES=TeacherForcedScoring
export VLLM_PLUGINS=teacher_forced_scoring
export OMP_NUM_THREADS=8
nvidia-smi > "$run_dir/gpu.txt"
vllm/.venv/bin/python -m pip freeze > "$run_dir/environment.txt"
# Native and HF encoders/decoders each run in a separate GPU process.
vllm/.venv/bin/python vllm/benchmarks/benchmark_autoregressive.py --output "$run_dir" "$@"
cat "$run_dir/comparison.md"
