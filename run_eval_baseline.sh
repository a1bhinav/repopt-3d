#!/usr/bin/env bash
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=3:59:00
#SBATCH --job-name=eval_baseline
#SBATCH --output=out/slurm-eval_baseline-%j.out
#SBATCH --error=out/slurm-eval_baseline-%j.err

# Baseline fusion mIoU on adapt-test (no training). Requires split_dataset + data env vars.
# Submit: sbatch run_eval_baseline.sh

set -euo pipefail
SCRIPT_DIR="${HOME}/Projects/NNProject/repopt-3d"
cd "${SCRIPT_DIR}"
mkdir -p "${SCRIPT_DIR}/out"

module load conda/latest
conda activate nnproject310

exec python "${SCRIPT_DIR}/scripts/eval_baseline.py" "$@"
