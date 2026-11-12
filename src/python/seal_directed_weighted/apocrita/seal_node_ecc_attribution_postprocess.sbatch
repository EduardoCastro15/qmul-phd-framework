#!/usr/bin/env bash
#SBATCH --job-name=seal-node-ecc-post
#SBATCH --partition=highmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail
: "${REPO_ROOT:?Set REPO_ROOT}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX}"
: "${POSTPROCESS_STAGE:?Set POSTPROCESS_STAGE to validate or summarize}"

WEIGHTED_DIR="${REPO_ROOT}/src/python/seal_directed_weighted"
PYTHON_BIN="${SEAL_ENV_PREFIX}/bin/python"
MANIFEST_PATH="${RUN_ROOT}/manifest/manifest.csv"
RUN_CONFIG_PATH="${RUN_ROOT}/manifest/RUN_CONFIG.json"
ATTRIBUTION_CONFIG_PATH="${RUN_ROOT}/manifest/ATTRIBUTION_CONFIG.json"

module load miniforge
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/seal-attribution-mpl-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${POSTPROCESS_STAGE}" == "validate" ]]; then
  ARGS=(
    --manifest "${MANIFEST_PATH}"
    --run-config "${RUN_CONFIG_PATH}"
    --attribution-config "${ATTRIBUTION_CONFIG_PATH}"
    --result-root "${RUN_ROOT}"
    --require-cuda --write-report
  )
  for job_file in \
    "${RUN_ROOT}/slurm/train-experiment1_job_id.txt" \
    "${RUN_ROOT}/slurm/train-campaign_job_id.txt"; do
    [[ ! -f "${job_file}" ]] || ARGS+=(
      --training-job-id "$(tr -d '[:space:]' < "${job_file}")"
    )
  done
  for job_file in \
    "${RUN_ROOT}/slurm/attr-experiment1_job_id.txt" \
    "${RUN_ROOT}/slurm/attr-campaign_job_id.txt"; do
    [[ ! -f "${job_file}" ]] || ARGS+=(
      --attribution-job-id "$(tr -d '[:space:]' < "${job_file}")"
    )
  done
  "${PYTHON_BIN}" "${WEIGHTED_DIR}/validate_attribution_campaign.py" "${ARGS[@]}"
elif [[ "${POSTPROCESS_STAGE}" == "summarize" ]]; then
  VALIDATION_REPORT="${RUN_ROOT}/summary/ATTRIBUTION_VALIDATION.json"
  "${PYTHON_BIN}" -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "VALIDATION_OK"' \
    "${VALIDATION_REPORT}"
  "${PYTHON_BIN}" "${WEIGHTED_DIR}/summarize_attributions.py" \
    --manifest "${MANIFEST_PATH}" --result-root "${RUN_ROOT}" \
    --output-dir "${RUN_ROOT}/summary/attribution"
else
  echo "Unknown POSTPROCESS_STAGE=${POSTPROCESS_STAGE}" >&2
  exit 2
fi
