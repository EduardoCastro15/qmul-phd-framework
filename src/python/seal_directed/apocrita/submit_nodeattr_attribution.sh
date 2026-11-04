#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ ! "${MODE}" =~ ^(smoke|calibration|experiment1|campaign|sanity)$ ]]; then
  echo "Usage: $0 smoke|calibration|experiment1|campaign|sanity [32|64|128|256]" >&2
  exit 2
fi
: "${RUN_ROOT:?Set RUN_ROOT to the immutable campaign root}"
: "${REPO_ROOT:?Set REPO_ROOT to the deployed checkout}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX}"
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT}"

MANIFEST_PATH="${RUN_ROOT}/manifest/manifest.csv"
CONFIG_PATH="${RUN_ROOT}/manifest/RUN_CONFIG.json"
TRAINING_RESULTS_ROOT="${TRAINING_RESULTS_ROOT:-${RUN_ROOT}}"
ATTRIBUTION_RESULTS_ROOT="${RUN_ROOT}"
SANITY_RANDOMIZE_THROUGH="${SANITY_RANDOMIZE_THROUGH:-}"

case "${MODE}" in
  calibration)
    SETTING="${2:-}"
    if [[ ! "${SETTING}" =~ ^(32|64|128|256)$ ]]; then
      echo "Calibration requires one setting: 32, 64, 128 or 256" >&2
      exit 2
    fi
    ATTRIBUTION_CONFIG_PATH="${RUN_ROOT}/calibration/config_${SETTING}/ATTRIBUTION_CONFIG.json"
    ATTRIBUTION_RESULTS_ROOT="${RUN_ROOT}/calibration/results_${SETTING}"
    ARRAY="1%1"
    ;;
  smoke)
    ATTRIBUTION_CONFIG_PATH="${RUN_ROOT}/manifest/ATTRIBUTION_CONFIG.json"
    ARRAY="1%1"
    ;;
  experiment1)
    ATTRIBUTION_CONFIG_PATH="${RUN_ROOT}/manifest/ATTRIBUTION_CONFIG.json"
    ARRAY="1%1"
    ;;
  campaign)
    ATTRIBUTION_CONFIG_PATH="${RUN_ROOT}/manifest/ATTRIBUTION_CONFIG.json"
    CALIBRATION_REPORT="${RUN_ROOT}/manifest/CALIBRATION_REPORT.json"
    if [[ ! -f "${CALIBRATION_REPORT}" ]]; then
      echo "Missing calibration report; ExperimentID 2-10 cannot be released." >&2
      exit 2
    fi
    "${SEAL_ENV_PREFIX}/bin/python" -c \
      'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "CALIBRATION_OK"' \
      "${CALIBRATION_REPORT}"
    ARRAY="2-10%${SEAL_ATTR_MAX_CONCURRENT:-2}"
    ;;
  sanity)
    ATTRIBUTION_CONFIG_PATH="${RUN_ROOT}/sanity/ATTRIBUTION_CONFIG.json"
    ATTRIBUTION_RESULTS_ROOT="${RUN_ROOT}/sanity/${SANITY_RANDOMIZE_THROUGH:-valid}"
    ARRAY="1%1"
    ;;
esac

for path in "${MANIFEST_PATH}" "${CONFIG_PATH}" "${ATTRIBUTION_CONFIG_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing immutable input: ${path}" >&2
    exit 2
  fi
done

SCRIPT="${REPO_ROOT}/src/python/seal_directed/apocrita/seal_nodeattr_attribution_array.sbatch"
mkdir -p "${ATTRIBUTION_RESULTS_ROOT}/slurm"
SUBMIT_ARGS=(
  --parsable
  --array="${ARRAY}"
  --export=ALL
  --output="${ATTRIBUTION_RESULTS_ROOT}/slurm/%x.o%A.%a"
)
if [[ "${SUBMIT_HELD:-1}" == "1" ]]; then
  SUBMIT_ARGS+=(--hold)
fi
export MANIFEST_PATH CONFIG_PATH ATTRIBUTION_CONFIG_PATH
export TRAINING_RESULTS_ROOT ATTRIBUTION_RESULTS_ROOT
export REPO_ROOT SEAL_ENV_PREFIX SCRATCH_ROOT SANITY_RANDOMIZE_THROUGH
JOB_ID="$(sbatch "${SUBMIT_ARGS[@]}" "${SCRIPT}")"
printf '%s\n' "${JOB_ID}" > "${ATTRIBUTION_RESULTS_ROOT}/slurm/${MODE}_job_id.txt"
echo "[INFO] Submitted ${MODE} attribution job ${JOB_ID} held=${SUBMIT_HELD:-1}"
