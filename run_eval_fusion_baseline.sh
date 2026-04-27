#!/usr/bin/env bash
# Baseline evaluation using OpenScene fusion mode only.

set -euo pipefail

SCRIPT_DIR="${HOME}/Projects/NNProject/repopt-3d"
OPENSE="${SCRIPT_DIR}/third_party/openscene"
DATA_ROOT="${DATA_ROOT:-/project/pi_hongyu_umass_edu/mdhassan/3d}"
DATA_3D="${DATA_3D:-${DATA_ROOT}/matterport_3d}"
DATA_FEAT="${DATA_FEAT:-${DATA_ROOT}/matterport_multiview_openseg_test}"
SPLIT="${SPLIT:-test}"
TEST_REPEATS="${TEST_REPEATS:-1}"
OUT_DIR="${OUT_DIR:-${OPENSE}/out/matterport_fusion_baseline_eval}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

mkdir -p "${OUT_DIR}"
cd "${OPENSE}"
export PYTHONPATH=.

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${OUT_DIR}/result_eval_fusion_${SPLIT}"
LOG="${OUT_DIR}/eval-${STAMP}_fusion_${SPLIT}.log"

echo "=========================================="
echo "fusion baseline evaluation"
echo "data_root=${DATA_3D}"
echo "fused_feature_root=${DATA_FEAT}"
echo "split=${SPLIT}"
echo "test_repeats=${TEST_REPEATS}"
echo "log=${LOG}"
echo "=========================================="

python -u run/evaluate.py \
  --config config/matterport/ours_openseg_pretrained.yaml \
  data_root "${DATA_3D}" \
  data_root_2d_fused_feature "${DATA_FEAT}" \
  feature_type "fusion" \
  vis_input "False" \
  vis_pred "False" \
  vis_gt "False" \
  split "${SPLIT}" \
  test_repeats "${TEST_REPEATS}" \
  save_folder "${RESULT_DIR}" \
  ${EXTRA_OPTS} 2>&1 | tee -a "${LOG}"
