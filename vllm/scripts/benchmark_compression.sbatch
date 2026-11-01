#!/usr/bin/env bash
#SBATCH --job-name=vllm-ac-bench
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=vllm-ac-bench-%j.log
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
run_dir="$PWD/vllm/results/compression-${SLURM_JOB_ID}"
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$run_dir/exit-code.txt"' EXIT
mkdir -p "$run_dir/source"
cp -r vllm/vllm_ac "$run_dir/source/"
cp -r vllm/benchmarks "$run_dir/source/"
export PYTHONPATH="$run_dir/source${PYTHONPATH:+:$PYTHONPATH}"
cp vllm/scripts/benchmark_compression.sbatch "$run_dir/source/"
export PYTHONUNBUFFERED=1 VLLM_CUSTOM_PATCHES=TeacherForcedScoring
export VLLM_PLUGINS=teacher_forced_scoring VLLM_AC_ENABLE=1 OMP_NUM_THREADS=8
nvidia-smi > "$run_dir/gpu.txt"
vllm/.venv/bin/python -m pip freeze > "$run_dir/environment.txt"
# Pass --input PATH to run an entire UTF-8 file; default is the multilingual smoke workload.
vllm/.venv/bin/python "$run_dir/source/benchmarks/benchmark_compression.py" --backend vllm --output "$run_dir/vllm.json" "$@"
vllm/.venv/bin/python "$run_dir/source/benchmarks/benchmark_compression.py" --backend hf --output "$run_dir/hf.json" "$@"
vllm/.venv/bin/python "$run_dir/source/benchmarks/benchmark_compression.py" --report "$run_dir"
vllm/.venv/bin/python "$run_dir/source/benchmarks/bundle_results.py" "$run_dir"
