#!/usr/bin/env bash
#SBATCH --partition=gpu-preempt
#SBATCH --qos=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=3:59:00
#SBATCH --job-name=h1_fusion_train
#SBATCH --output=third_party/openscene/out/slurm-%j.out
#SBATCH --error=third_party/openscene/out/slurm-%j.err

# Train H1 adapter on fusion features only (no distill / no 3D network).
# Strict reproduction mode:
# - use legacy split names: train/val
# - deterministic 50/50 split from test

module load conda/latest

conda activate nnproject310

set -euo pipefail

SCRIPT_DIR="${HOME}/Projects/NNProject/repopt-3d"
OPENSE="${SCRIPT_DIR}/third_party/openscene"
DATA_ROOT="${DATA_ROOT:-/project/pi_hongyu_umass_edu/mdhassan/3d}"
SRC_3D="${SRC_3D:-${DATA_ROOT}/matterport_3d}"
SRC_FEAT="${SRC_FEAT:-${DATA_ROOT}/matterport_multiview_openseg_test}"

SEED="${SEED:-3407}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATA_ROOT}/matterport_3d_h1split_seed${SEED}}"
SAVE_PATH="${SAVE_PATH:-${OPENSE}/out/matterport_h1_fusion_seed${SEED}}"
CONFIG_PATH="${CONFIG_PATH:-${OPENSE}/config/matterport/h1_fusion.yaml}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

if [[ ! -d "${SRC_3D}/test" ]]; then
  echo "Missing source test dir: ${SRC_3D}/test" >&2
  exit 1
fi
if [[ ! -d "${SRC_FEAT}" ]]; then
  echo "Missing fused feature dir: ${SRC_FEAT}" >&2
  exit 1
fi

mkdir -p "${SPLIT_ROOT}/train" "${SPLIT_ROOT}/val"
rm -f "${SPLIT_ROOT}/train/"*.pth "${SPLIT_ROOT}/val/"*.pth || true

python - "${SRC_3D}" "${SPLIT_ROOT}" "${SEED}" <<'PY'
import os
import random
import sys
from pathlib import Path

src_root = Path(sys.argv[1])
split_root = Path(sys.argv[2])
seed = int(sys.argv[3])
src_test = src_root / "test"
train_dir = split_root / "train"
val_dir = split_root / "val"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

scenes = sorted([p for p in src_test.glob("*.pth")])
if len(scenes) < 2:
    raise RuntimeError(f"Need >=2 test scenes, found {len(scenes)}")
random.Random(seed).shuffle(scenes)
n_adapt = max(1, int(round(len(scenes) * 0.5)))
n_adapt = min(n_adapt, len(scenes) - 1)
split_adapt = scenes[:n_adapt]
split_heldout = scenes[n_adapt:]

for p in split_adapt:
    target = train_dir / p.name
    target.symlink_to(p)
for p in split_heldout:
    target = val_dir / p.name
    target.symlink_to(p)

print(f"Built split at {split_root}")
print(f"train scenes: {len(split_adapt)}")
print(f"val scenes:   {len(split_heldout)}")
PY

mkdir -p "${SAVE_PATH}"
cd "${OPENSE}"
export PYTHONPATH=.

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${SAVE_PATH}/train-h1-fusion-${STAMP}.log"

echo "=========================================="
echo "H1 fusion-only training"
echo "split_root=${SPLIT_ROOT}"
echo "feat_root=${SRC_FEAT}"
echo "save_path=${SAVE_PATH}"
echo "seed=${SEED}"
echo "log=${LOG}"
echo "=========================================="

python -u run/train_fusion_h1.py \
  --config "${CONFIG_PATH}" \
  data_root "${SPLIT_ROOT}" \
  adapt_split "train" \
  heldout_split "val" \
  data_root_2d_fused_feature "${SRC_FEAT}" \
  save_path "${SAVE_PATH}" \
  manual_seed "${SEED}" \
  ${EXTRA_OPTS} 2>&1 | tee -a "${LOG}"
