#!/usr/bin/env bash
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --job-name=split_dataset
#SBATCH --output=out/slurm-split_dataset-%j.out
#SBATCH --error=out/slurm-split_dataset-%j.err

# Build adapt-train / adapt-val / adapt-test symlinks from matterport_3d/test.
# Submit: sbatch split_dataset.sh

set -euo pipefail
SCRIPT_DIR="${HOME}/Projects/NNProject/repopt-3d"
cd "${SCRIPT_DIR}"
mkdir -p "${SCRIPT_DIR}/out"

module load conda/latest
conda activate nnproject310

exec python "${SCRIPT_DIR}/scripts/split_dataset.py" "$@"
