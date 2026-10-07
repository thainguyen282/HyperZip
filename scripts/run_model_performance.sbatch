#!/usr/bin/env bash
# Submit from the HyperZip repository root: sbatch scripts/run_model_performance.sbatch
# Each array task runs one model on the same FineWeb validation input.
# Each submission writes to its own results directory.
#SBATCH --job-name=hyperzip-models
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=scripts/model-performance-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the HyperZip repository root}"

case "${SLURM_ARRAY_TASK_ID:?}" in
  0)
    name=fastdllm15_base
    python=.venv/bin/python
    trust=(--trust_remote_code)
    ;;
  1)
    name=nemotron3b_base
    python=attempt/.automodel/.venv/bin/python
    trust=(--trust_remote_code)
    ;;
  2)
    name=qwen25_15b_base
    python=.venv/bin/python
    trust=()
    ;;
  *)
    echo "Unknown array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

stem="fineweb_validation__${name}"
input=experiments/inputs/fineweb_validation.txt
template="experiments/performance_configs/${stem}.yaml"
run_dir="results/experiments/performance_runs/${SLURM_ARRAY_JOB_ID}/${stem}"
mkdir -p "$run_dir"
config="$run_dir/config.yaml"
archive="$run_dir/archive.bin"
restored="$run_dir/restored.txt"
sed "s|^output: .*|output: $archive|" "$template" > "$config"

"$python" main.py --config "$config"
"$python" main.py --step decode --input "$archive" --output "$restored" \
  --device cuda "${trust[@]}" --no_progress
cmp "$input" "$restored"
echo "VERIFIED_BYTE_EQUALITY $stem"
