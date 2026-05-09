#!/usr/bin/env bash
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --job-name=hpo_ce
#SBATCH --output=out/slurm-hpo_ce-%j.out
#SBATCH --error=out/slurm-hpo_ce-%j.err

# HPO with CE-only adapter; final metric on adapt-test. Run split_dataset.py first.
# Submit: sbatch run_hpo_ce.sh

set -euo pipefail
SCRIPT_DIR="${HOME}/Projects/NNProject/repopt-3d"
cd "${SCRIPT_DIR}"
mkdir -p "${SCRIPT_DIR}/out"

module load conda/latest
conda activate nnproject310

exec python "${SCRIPT_DIR}/scripts/hpo_fusion.py" --loss-mode ce "$@"
