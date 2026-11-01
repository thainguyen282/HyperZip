#!/usr/bin/env bash
#SBATCH --job-name=vllm-codec-validation
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=vllm-codec-validation-%j.log
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the HyperZip repository root}"

run_dir="$PWD/vllm/results/${SLURM_JOB_ID}"
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$run_dir/exit-code.txt"' EXIT
export PYTHONUNBUFFERED=1
export VLLM_CUSTOM_PATCHES=TeacherForcedScoring
export VLLM_PLUGINS=teacher_forced_scoring
export VLLM_AC_RESULT_DIR="$run_dir"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_AC_ENABLE=0
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8

# Serialize environment setup if multiple validation jobs start together.
(
    flock -x 9
    if [[ ! -x vllm/.venv/bin/python ]]; then
        .venv/bin/python -m venv vllm/.venv
    fi
    vllm/.venv/bin/python -m pip install -e ./vllm
) 9>vllm/.setup.lock

nvidia-smi
vllm/.venv/bin/python -m pip freeze > "$run_dir/environment.txt"
vllm/.venv/bin/python -m unittest discover -s vllm/tests -v
export RUN_VLLM_AC_AR_GPU_TESTS=1
vllm/.venv/bin/python -m unittest discover -s vllm/tests -p test_autoregressive_gpu.py -v
