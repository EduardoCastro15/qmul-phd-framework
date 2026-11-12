#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ ! "${MODE}" =~ ^(init|train-smoke|train-experiment1|train-campaign|attr-smoke|calibration|freeze|attr-experiment1|attr-campaign|validate|summarize)$ ]]; then
  echo "Usage: $0 init|train-smoke|train-experiment1|train-campaign|attr-smoke|calibration [32|64|128|256]|freeze|attr-experiment1|attr-campaign|validate|summarize" >&2
  exit 2
fi
: "${REPO_ROOT:?Set REPO_ROOT to the deployed checkout}"
: "${RUN_ROOT:?Set RUN_ROOT to a new persistent SEAL_directed_node_ecc_attribution_v2 root}"
: "${HISTORICAL_RUN_ROOT:?Set HISTORICAL_RUN_ROOT to the completed four-variant campaign}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX}"
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT under /gpfs/scratch/$USER}"

WEIGHTED_DIR="${REPO_ROOT}/src/python/seal_directed_weighted"
PYTHON_BIN="${SEAL_ENV_PREFIX}/bin/python"
MANIFEST_DIR="${RUN_ROOT}/manifest"
MANIFEST_PATH="${MANIFEST_DIR}/manifest.csv"
RUN_CONFIG_PATH="${MANIFEST_DIR}/RUN_CONFIG.json"
MANIFEST_LOCK_PATH="${MANIFEST_DIR}/MANIFEST_LOCK.json"
SCRIPT="${WEIGHTED_DIR}/apocrita/seal_node_ecc_attribution_array.sbatch"
POSTPROCESS_SCRIPT="${WEIGHTED_DIR}/apocrita/seal_node_ecc_attribution_postprocess.sbatch"
test -x "${PYTHON_BIN}" || { echo "Missing Python executable: ${PYTHON_BIN}" >&2; exit 2; }
if "${PYTHON_BIN}" -c \
  'from pathlib import Path; import sys; raise SystemExit(Path(sys.argv[1]).resolve() != Path(sys.argv[2]).resolve())' \
  "${RUN_ROOT}" "${HISTORICAL_RUN_ROOT}"; then
  echo "RUN_ROOT must be different from HISTORICAL_RUN_ROOT; historical results are immutable." >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}/slurm"
"${PYTHON_BIN}" -c \
  'import captum; assert captum.__version__ == "0.9.0", captum.__version__' || {
    echo "The selected environment must contain captum==0.9.0." >&2
    exit 2
  }
if [[ -f "${REPO_ROOT}/SOURCE_COMMIT.txt" ]]; then
  SOURCE_COMMIT="$(tr -d '[:space:]' < "${REPO_ROOT}/SOURCE_COMMIT.txt")"
else
  SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
fi
if [[ "${MODE}" == "init" && ! -f "${REPO_ROOT}/DEPLOYMENT_MANIFEST.sha256" ]]; then
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain -- \
    src/python/seal_directed_weighted src/python/pytorch_DGCNN)" ]]; then
    echo "Refusing an uncommitted source snapshot; commit or deploy a checksummed bundle first." >&2
    exit 2
  fi
fi

if [[ "${MODE}" == "init" ]]; then
  test ! -e "${MANIFEST_DIR}" || { echo "Manifest already exists: ${MANIFEST_DIR}" >&2; exit 2; }
  "${PYTHON_BIN}" "${WEIGHTED_DIR}/generate_attribution_manifest.py" \
    --source-manifest "${HISTORICAL_RUN_ROOT}/manifest/manifest.csv" \
    --source-config "${HISTORICAL_RUN_ROOT}/manifest/RUN_CONFIG.json" \
    --output-dir "${MANIFEST_DIR}" --expected-foodwebs 290 --expected-experiments 10 \
    --source-commit "${SOURCE_COMMIT}"
  for setting in 32 64 128 256; do
    "${PYTHON_BIN}" "${WEIGHTED_DIR}/generate_attribution_config.py" \
      --output "${RUN_ROOT}/calibration/config_${setting}/ATTRIBUTION_CONFIG.json" \
      --setting "${setting}" --faithfulness-background-samples 32
  done
  mkdir -p "${RUN_ROOT}/environment"
  cp "${WEIGHTED_DIR}/apocrita/environment.apocrita.yml" "${RUN_ROOT}/environment/"
  cp "${WEIGHTED_DIR}/apocrita/requirements-apocrita.txt" "${RUN_ROOT}/environment/"
  if [[ -f "${SEAL_ENV_PREFIX}/pip-freeze.txt" ]]; then
    cp "${SEAL_ENV_PREFIX}/pip-freeze.txt" "${RUN_ROOT}/environment/"
  fi
  printf '%s\n' "${SOURCE_COMMIT}" > "${RUN_ROOT}/SOURCE_COMMIT.txt"
  if [[ -f "${REPO_ROOT}/DEPLOYMENT_MANIFEST.sha256" ]]; then
    cp "${REPO_ROOT}/DEPLOYMENT_MANIFEST.sha256" "${RUN_ROOT}/"
    [[ ! -f "${REPO_ROOT}/BUNDLE_METADATA.json" ]] || cp "${REPO_ROOT}/BUNDLE_METADATA.json" "${RUN_ROOT}/"
  else
    git -C "${REPO_ROOT}" archive --format=tar.gz \
      --output="${RUN_ROOT}/source_snapshot.tar.gz" HEAD -- \
      src/python/seal_directed_weighted \
      src/python/pytorch_DGCNN \
      ':(exclude)src/python/seal_directed_weighted/data'
  fi
  echo "[OK] Initialized ${RUN_ROOT}. No jobs submitted."
  exit 0
fi

for path in "${MANIFEST_PATH}" "${RUN_CONFIG_PATH}" "${MANIFEST_LOCK_PATH}" "${SCRIPT}" "${POSTPROCESS_SCRIPT}"; do
  test -f "${path}" || { echo "Missing ${path}; run init first." >&2; exit 2; }
done
"${PYTHON_BIN}" "${WEIGHTED_DIR}/attribution_manifest_lock.py" \
  --manifest "${MANIFEST_PATH}" --config "${RUN_CONFIG_PATH}" \
  --lock "${MANIFEST_LOCK_PATH}"
if [[ "${MODE}" == "freeze" ]]; then
  "${PYTHON_BIN}" "${WEIGHTED_DIR}/freeze_attribution_config.py" \
    --manifest "${MANIFEST_PATH}" --output-dir "${MANIFEST_DIR}" \
    --calibration-root "32=${RUN_ROOT}/calibration/results_32" \
    --calibration-root "64=${RUN_ROOT}/calibration/results_64" \
    --calibration-root "128=${RUN_ROOT}/calibration/results_128" \
    --calibration-root "256=${RUN_ROOT}/calibration/results_256" \
    --expected-foodwebs 290
  echo "[OK] Calibration frozen in ${MANIFEST_DIR}"
  exit 0
fi
if [[ "${MODE}" == "validate" || "${MODE}" == "summarize" ]]; then
  ATTRIBUTION_CONFIG_PATH="${MANIFEST_DIR}/ATTRIBUTION_CONFIG.json"
  test -f "${ATTRIBUTION_CONFIG_PATH}" || {
    echo "Missing frozen ${ATTRIBUTION_CONFIG_PATH}" >&2; exit 2;
  }
  if [[ "${MODE}" == "summarize" ]]; then
    VALIDATION_REPORT="${RUN_ROOT}/summary/ATTRIBUTION_VALIDATION.json"
    "${PYTHON_BIN}" -c \
      'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "VALIDATION_OK"' \
      "${VALIDATION_REPORT}"
  fi
  export REPO_ROOT RUN_ROOT SEAL_ENV_PREFIX POSTPROCESS_STAGE="${MODE}"
  POST_SUBMIT=(--parsable \
    --partition="${SEAL_POSTPROCESS_PARTITION:-highmem}" \
    --time="${SEAL_POSTPROCESS_TIME:-24:00:00}" \
    --cpus-per-task="${SEAL_POSTPROCESS_CPUS:-4}" \
    --mem="${SEAL_POSTPROCESS_MEMORY:-128G}" \
    --job-name="seal-v2-${MODE}" --export=ALL \
    --output="${RUN_ROOT}/slurm/%x.o%j" \
    --error="${RUN_ROOT}/slurm/%x.e%j")
  if [[ "${SUBMIT_HELD:-1}" == "1" ]]; then POST_SUBMIT+=(--hold); fi
  POST_JOB_ID="$(sbatch "${POST_SUBMIT[@]}" "${POSTPROCESS_SCRIPT}")"
  printf '%s\n' "${POST_JOB_ID}" > "${RUN_ROOT}/slurm/${MODE}_job_id.txt"
  echo "[INFO] Submitted ${MODE} job ${POST_JOB_ID} in HOLD=${SUBMIT_HELD:-1}"
  echo "[INFO] Inspect, then release with: scontrol release ${POST_JOB_ID}"
  exit 0
fi
export REPO_ROOT SEAL_ENV_PREFIX SCRATCH_ROOT HISTORICAL_RUN_ROOT SOURCE_COMMIT
export MANIFEST_PATH RUN_CONFIG_PATH TRAINING_RESULT_ROOT="${RUN_ROOT}"
export SOURCE_COMMIT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "${RUN_CONFIG_PATH}")"
unset ONLY_FOODWEB ATTRIBUTION_CONFIG_PATH ATTRIBUTION_RESULT_ROOT

CONFIG_FOODWEBS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_foodwebs"])' "${RUN_CONFIG_PATH}")"
CONFIG_EXPERIMENTS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_experiments"])' "${RUN_CONFIG_PATH}")"
[[ "${CONFIG_FOODWEBS}" == "290" && "${CONFIG_EXPERIMENTS}" == "10" ]] || {
  echo "This production launcher requires the frozen 290-food-web x 10-experiment manifest." >&2
  exit 2
}
FOODWEB_BLOCK_SIZE="${SEAL_FOODWEB_BLOCK_SIZE:-6}"
[[ "${FOODWEB_BLOCK_SIZE}" =~ ^[0-9]+$ ]] && (( FOODWEB_BLOCK_SIZE >= 6 )) || {
  echo "SEAL_FOODWEB_BLOCK_SIZE must be an integer >= 6." >&2
  exit 2
}
BLOCKS_PER_SHARD=$(( (CONFIG_FOODWEBS + FOODWEB_BLOCK_SIZE - 1) / FOODWEB_BLOCK_SIZE ))
MAX_ARRAY_INDEX=$(( 2 * CONFIG_EXPERIMENTS * BLOCKS_PER_SHARD ))
(( MAX_ARRAY_INDEX <= 1000 )) || {
  echo "Block mapping would require array index ${MAX_ARRAY_INDEX} (>1000)." >&2
  exit 2
}
NODE_EXP1_RANGE="1-${BLOCKS_PER_SHARD}"
NODE_CAMPAIGN_RANGE="$((BLOCKS_PER_SHARD + 1))-$((CONFIG_EXPERIMENTS * BLOCKS_PER_SHARD))"
ECC_EXP1_RANGE="$((CONFIG_EXPERIMENTS * BLOCKS_PER_SHARD + 1))-$(((CONFIG_EXPERIMENTS + 1) * BLOCKS_PER_SHARD))"
ECC_CAMPAIGN_RANGE="$(((CONFIG_EXPERIMENTS + 1) * BLOCKS_PER_SHARD + 1))-${MAX_ARRAY_INDEX}"
SMOKE_ECC_TASK=$((CONFIG_EXPERIMENTS + 1))

PIPELINE_STAGE="training"
TASK_SELECTION="shard"
ARRAY=""; NAME=""; ATTRIBUTION_CONFIG_PATH=""; ATTRIBUTION_RESULT_ROOT="${RUN_ROOT}"
case "${MODE}" in
  train-smoke)
    ARRAY="1,${SMOKE_ECC_TASK}%1"; NAME="seal-v2-train-smoke"; ONLY_FOODWEB="${SMOKE_FOODWEB:-SF1M2_tax_mass}" ;;
  train-experiment1)
    TASK_SELECTION="block"; ARRAY="${NODE_EXP1_RANGE},${ECC_EXP1_RANGE}%${SEAL_MAX_CONCURRENT:-2}"; NAME="seal-v2-train-exp1" ;;
  train-campaign)
    "${PYTHON_BIN}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "CALIBRATION_OK"' "${MANIFEST_DIR}/CALIBRATION_REPORT.json"
    TASK_SELECTION="block"; ARRAY="${NODE_CAMPAIGN_RANGE},${ECC_CAMPAIGN_RANGE}%${SEAL_MAX_CONCURRENT:-2}"; NAME="seal-v2-train-full" ;;
  attr-smoke)
    PIPELINE_STAGE="attribution"; ARRAY="1,${SMOKE_ECC_TASK}%1"; NAME="seal-v2-attr-smoke"
    ONLY_FOODWEB="${SMOKE_FOODWEB:-SF1M2_tax_mass}"
    ATTRIBUTION_CONFIG_PATH="${RUN_ROOT}/calibration/config_32/ATTRIBUTION_CONFIG.json"
    ATTRIBUTION_RESULT_ROOT="${RUN_ROOT}/smoke" ;;
  calibration)
    SETTING="${2:-}"; [[ "${SETTING}" =~ ^(32|64|128|256)$ ]] || { echo "calibration requires 32|64|128|256" >&2; exit 2; }
    PIPELINE_STAGE="attribution"; TASK_SELECTION="block"
    ARRAY="${NODE_EXP1_RANGE},${ECC_EXP1_RANGE}%${SEAL_MAX_CONCURRENT:-2}"; NAME="seal-v2-cal-${SETTING}"
    ATTRIBUTION_CONFIG_PATH="${RUN_ROOT}/calibration/config_${SETTING}/ATTRIBUTION_CONFIG.json"
    ATTRIBUTION_RESULT_ROOT="${RUN_ROOT}/calibration/results_${SETTING}" ;;
  attr-experiment1)
    PIPELINE_STAGE="attribution"; TASK_SELECTION="block"
    ARRAY="${NODE_EXP1_RANGE},${ECC_EXP1_RANGE}%${SEAL_MAX_CONCURRENT:-2}"; NAME="seal-v2-attr-exp1"
    ATTRIBUTION_CONFIG_PATH="${MANIFEST_DIR}/ATTRIBUTION_CONFIG.json" ;;
  attr-campaign)
    "${PYTHON_BIN}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "CALIBRATION_OK"' "${MANIFEST_DIR}/CALIBRATION_REPORT.json"
    PIPELINE_STAGE="attribution"; TASK_SELECTION="block"
    ARRAY="${NODE_CAMPAIGN_RANGE},${ECC_CAMPAIGN_RANGE}%${SEAL_MAX_CONCURRENT:-2}"; NAME="seal-v2-attr-full"
    ATTRIBUTION_CONFIG_PATH="${MANIFEST_DIR}/ATTRIBUTION_CONFIG.json" ;;
esac
if [[ "${PIPELINE_STAGE}" == "attribution" ]]; then
  test -f "${ATTRIBUTION_CONFIG_PATH}" || {
    echo "Missing attribution config: ${ATTRIBUTION_CONFIG_PATH}" >&2
    exit 2
  }
fi
export PIPELINE_STAGE TASK_SELECTION FOODWEB_BLOCK_SIZE ONLY_FOODWEB ATTRIBUTION_CONFIG_PATH ATTRIBUTION_RESULT_ROOT
SETTING="${SETTING:-}"
mkdir -p "${ATTRIBUTION_RESULT_ROOT}/slurm"
if [[ "${MODE}" == "train-smoke" || "${MODE}" == "attr-smoke" ]]; then
  RESOURCES=(
    --partition="${SEAL_SMOKE_PARTITION:-gpushort}"
    --time="${SEAL_SMOKE_TIME:-01:00:00}"
  )
else
  RESOURCES=(
    --account="${SEAL_GPU_ACCOUNT:-pilot_sae_gpu}"
    --partition="${SEAL_GPU_PARTITION:-sae}"
    --constraint="${SEAL_GPU_CONSTRAINT:-ampere}"
    --time="${SEAL_GPU_TIME:-24:00:00}"
  )
fi
SUBMIT=("${RESOURCES[@]}" --parsable --array="${ARRAY}" --job-name="${NAME}" --export=ALL
        --output="${ATTRIBUTION_RESULT_ROOT}/slurm/%x.o%A.%a"
        --error="${ATTRIBUTION_RESULT_ROOT}/slurm/%x.e%A.%a")
if [[ "${SUBMIT_HELD:-1}" == "1" ]]; then SUBMIT+=(--hold); fi
JOB_ID="$(sbatch "${SUBMIT[@]}" "${SCRIPT}")"
printf '%s\n' "${JOB_ID}" > "${ATTRIBUTION_RESULT_ROOT}/slurm/${MODE}${SETTING:+_${SETTING}}_job_id.txt"
{
  printf 'mode=%s\n' "${MODE}"
  printf 'pipeline_stage=%s\n' "${PIPELINE_STAGE}"
  printf 'task_selection=%s\n' "${TASK_SELECTION}"
  printf 'foodweb_block_size=%s\n' "${FOODWEB_BLOCK_SIZE}"
  printf 'blocks_per_shard=%s\n' "${BLOCKS_PER_SHARD}"
  printf 'array=%s\n' "${ARRAY}"
  printf 'source_commit=%s\n' "${SOURCE_COMMIT}"
  printf 'training_result_root=%s\n' "${TRAINING_RESULT_ROOT}"
  printf 'attribution_result_root=%s\n' "${ATTRIBUTION_RESULT_ROOT}"
  printf 'attribution_config=%s\n' "${ATTRIBUTION_CONFIG_PATH:-none}"
  printf 'held=%s\n' "${SUBMIT_HELD:-1}"
  printf 'job_id=%s\n' "${JOB_ID}"
} > "${ATTRIBUTION_RESULT_ROOT}/slurm/${MODE}${SETTING:+_${SETTING}}_submission_config.txt"
echo "[INFO] Submitted ${MODE} job ${JOB_ID} in HOLD=${SUBMIT_HELD:-1}"
echo "[INFO] Inspect, then release with: scontrol release ${JOB_ID}"
