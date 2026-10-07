#!/usr/bin/env bash
#SBATCH --job-name=mlp-s-1536
#SBATCH --account=phan
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=results/hypernetwork-nemotron/mlp-s-1536-%j.log
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from HyperZip root}"
export EXPERIMENT_CONFIG=experiments/mlp_granularity_3b/mlp_s.yaml
exec /usr/bin/bash attempt/train_hypernetwork_3b_900m.sbatch
