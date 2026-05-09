#!/usr/bin/env bash
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=3:00:00
#SBATCH --job-name=eval_full_test
#SBATCH --output=out/slurm-eval_full_test-%j.out
#SBATCH --error=out/slurm-eval_full_test-%j.err

# Fusion baseline on full Matterport test split (406 scenes — longer than adapt-test).
# Submit: sbatch run_eval_full_test.sh

set -euo pipefail
SCRIPT_DIR="${HOME}/Projects/NNProject/repopt-3d"
cd "${SCRIPT_DIR}"
mkdir -p "${SCRIPT_DIR}/out"

module load conda/latest
conda activate nnproject310

exec python "${SCRIPT_DIR}/scripts/eval_full_test.py" "$@"
