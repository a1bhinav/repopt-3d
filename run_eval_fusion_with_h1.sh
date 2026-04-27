#!/usr/bin/env bash
# Evaluate fusion + learned H1 adapter (mode: fusion-with-h1).

set -euo pipefail

SCRIPT_DIR="${HOME}/Projects/NNProject/repopt-3d"
OPENSE="${SCRIPT_DIR}/third_party/openscene"
DATA_ROOT="${DATA_ROOT:-/project/pi_hongyu_umass_edu/mdhassan/3d}"
DATA_3D="${DATA_3D:-${DATA_ROOT}/matterport_3d}"
DATA_FEAT="${DATA_FEAT:-${DATA_ROOT}/matterport_multiview_openseg_test}"
OUT_DIR="${OUT_DIR:-${OPENSE}/out/matterport_fusion_with_h1_eval}"
SPLIT="${SPLIT:-test}"
TEST_REPEATS="${TEST_REPEATS:-1}"
MODEL_PATH="${MODEL_PATH:-${OPENSE}/out/matterport_h1_fusion_seed3407/model/model_best.pth.tar}"
ADAPTER_INPUT_DIM="${ADAPTER_INPUT_DIM:-768}"
ADAPTER_HIDDEN_DIM="${ADAPTER_HIDDEN_DIM:-512}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.1}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH to your trained adapter checkpoint." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cd "${OPENSE}"
export PYTHONPATH=.

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${OUT_DIR}/result_eval_fusion_with_h1_${SPLIT}"
LOG="${OUT_DIR}/eval-${STAMP}_fusion-with-h1_${SPLIT}.log"

echo "=========================================="
echo "fusion-with-h1 evaluation"
echo "data_root=${DATA_3D}"
echo "fused_feature_root=${DATA_FEAT}"
echo "model_path=${MODEL_PATH}"
echo "split=${SPLIT}"
echo "test_repeats=${TEST_REPEATS}"
echo "log=${LOG}"
echo "=========================================="

python -u run/evaluate.py \
  --config config/matterport/ours_openseg_pretrained.yaml \
  data_root "${DATA_3D}" \
  data_root_2d_fused_feature "${DATA_FEAT}" \
  feature_type "fusion-with-h1" \
  model_path "${MODEL_PATH}" \
  adapter_input_dim "${ADAPTER_INPUT_DIM}" \
  adapter_hidden_dim "${ADAPTER_HIDDEN_DIM}" \
  adapter_dropout "${ADAPTER_DROPOUT}" \
  vis_input "False" \
  vis_pred "False" \
  vis_gt "False" \
  split "${SPLIT}" \
  test_repeats "${TEST_REPEATS}" \
  save_folder "${RESULT_DIR}" \
  ${EXTRA_OPTS} 2>&1 | tee -a "${LOG}"
