#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "calibration" && "${MODE}" != "campaign" ]]; then
  echo "Usage: $0 {smoke|calibration|campaign}" >&2
  exit 2
fi
: "${REPO_ROOT:?Set REPO_ROOT to the clean deployed bundle}"
: "${PROJECT_RESULTS_ROOT:?Set PROJECT_RESULTS_ROOT to persistent storage}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX}"
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT under /gpfs/scratch/$USER}"

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
WEIGHTED_DIR="${REPO_ROOT}/src/python/seal_directed_weighted"
PYTHON_BIN="${SEAL_ENV_PREFIX}/bin/python"
RUN_ID="${RUN_ID:-seal_directed_weighted_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${PROJECT_RESULTS_ROOT}/${RUN_ID}"
MANIFEST_DIR="${RUN_ROOT}/manifest"
NUM_EXPERIMENTS="${SEAL_NUM_EXPERIMENTS:-10}"
MAX_CONCURRENT="${SEAL_MAX_CONCURRENT:-2}"

test -x "${PYTHON_BIN}" || { echo "[ERROR] Missing ${PYTHON_BIN}" >&2; exit 2; }
test -f "${REPO_ROOT}/src/python/pytorch_DGCNN/lib/build/dll/libgnn.so" || {
  echo "[ERROR] Build libgnn.so with /usr/bin/g++ before submission." >&2; exit 2;
}
if [[ -f "${REPO_ROOT}/DEPLOYMENT_MANIFEST.sha256" ]]; then
  "${PYTHON_BIN}" "${WEIGHTED_DIR}/apocrita/verify_bundle.py" --root "${REPO_ROOT}"
  SOURCE_COMMIT="$(tr -d '[:space:]' < "${REPO_ROOT}/SOURCE_COMMIT.txt")"
else
  SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain -- "${WEIGHTED_DIR}" "${REPO_ROOT}/src/python/pytorch_DGCNN")" ]] || {
    echo "[ERROR] Commit weighted bundle inputs first." >&2; exit 2;
  }
fi
mkdir -p "${SCRATCH_ROOT}" "${PROJECT_RESULTS_ROOT}"
if command -v qmquota >/dev/null 2>&1; then qmquota -s; fi

if [[ ! -e "${RUN_ROOT}" ]]; then
  mkdir -p "${MANIFEST_DIR}" "${RUN_ROOT}/slurm" "${RUN_ROOT}/environment"
  MANIFEST_ARGS=(
    --foodweb-csv "${REPO_ROOT}/src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
    --mat-folder "${WEIGHTED_DIR}/data/foodwebs_mat_seal_weighted"
    --output-dir "${MANIFEST_DIR}"
    --source-commit "${SOURCE_COMMIT}"
  )
  if [[ "${MODE}" == "smoke" ]]; then
    MANIFEST_ARGS+=(
      --num-experiments 1 --expected-foodwebs 3
      --only-foodweb AEW01_tax_mass
      --only-foodweb PP1I4_tax_mass
      --only-foodweb "Chesapeake Bay_tax_mass"
    )
  else
    MANIFEST_ARGS+=(--num-experiments "${NUM_EXPERIMENTS}" --expected-foodwebs 290)
  fi
  "${PYTHON_BIN}" "${WEIGHTED_DIR}/generate_manifest.py" "${MANIFEST_ARGS[@]}"
  printf '%s\n' "${SOURCE_COMMIT}" > "${RUN_ROOT}/SOURCE_COMMIT.txt"
  cp "${WEIGHTED_DIR}/apocrita/environment.apocrita.yml" "${RUN_ROOT}/environment/"
  cp "${WEIGHTED_DIR}/apocrita/requirements-apocrita.txt" "${RUN_ROOT}/environment/"
  [[ ! -f "${SEAL_ENV_PREFIX}/pip-freeze.txt" ]] || cp "${SEAL_ENV_PREFIX}/pip-freeze.txt" "${RUN_ROOT}/environment/"
  if [[ -f "${REPO_ROOT}/BUNDLE_METADATA.json" ]]; then
    cp "${REPO_ROOT}/BUNDLE_METADATA.json" "${RUN_ROOT}/"
    cp "${REPO_ROOT}/DEPLOYMENT_MANIFEST.sha256" "${RUN_ROOT}/"
    tar --exclude='*/__pycache__' --exclude='src/python/pytorch_DGCNN/lib/build' \
      -czf "${RUN_ROOT}/source_snapshot.tar.gz" -C "${REPO_ROOT}" .
  else
    git -C "${REPO_ROOT}" archive --format=tar.gz \
      --output="${RUN_ROOT}/source_snapshot.tar.gz" HEAD
  fi
else
  test -f "${MANIFEST_DIR}/manifest.csv" && test -f "${MANIFEST_DIR}/RUN_CONFIG.json" || {
    echo "[ERROR] Existing RUN_ROOT is incomplete." >&2; exit 2;
  }
  FROZEN_COMMIT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "${MANIFEST_DIR}/RUN_CONFIG.json")"
  [[ "${FROZEN_COMMIT}" == "${SOURCE_COMMIT}" ]] || {
    echo "[ERROR] RUN_CONFIG commit ${FROZEN_COMMIT} != deployment ${SOURCE_COMMIT}." >&2; exit 2;
  }
fi

CONFIG_EXPERIMENTS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_experiments"])' "${MANIFEST_DIR}/RUN_CONFIG.json")"
CONFIG_FOODWEBS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_foodwebs"])' "${MANIFEST_DIR}/RUN_CONFIG.json")"
export MANIFEST_PATH="${MANIFEST_DIR}/manifest.csv"
export CONFIG_PATH="${MANIFEST_DIR}/RUN_CONFIG.json"
export PROJECT_RESULTS_ROOT="${RUN_ROOT}"
export REPO_ROOT SEAL_ENV_PREFIX SCRATCH_ROOT SOURCE_COMMIT
SCRIPT="${WEIGHTED_DIR}/apocrita/seal_weighted_gpu_array.sbatch"
LOG_PATTERN="${RUN_ROOT}/slurm/%x.o%A.%a"

if [[ "${MODE}" == "smoke" ]]; then
  ARRAY_SPEC="1-4%1"
  SUBMIT=(--partition=gpushort --time=01:00:00 --job-name=seal-weighted-smoke)
elif [[ "${MODE}" == "calibration" ]]; then
  [[ "${CONFIG_EXPERIMENTS}" == "10" ]] || { echo "[ERROR] Calibration requires 10 experiments." >&2; exit 2; }
  ARRAY_SPEC="1,11,21,31%1"
  SUBMIT=(
    --account="${SEAL_GPU_ACCOUNT:-pilot_sae_gpu}"
    --partition="${SEAL_GPU_PARTITION:-sae}"
    --constraint="${SEAL_GPU_CONSTRAINT:-ampere}"
    --time="${SEAL_GPU_TIME:-24:00:00}"
    --job-name=seal-weighted-calibration
  )
else
  TOTAL_TASKS="$((4 * CONFIG_EXPERIMENTS))"
  ARRAY_SPEC="1-${TOTAL_TASKS}%${MAX_CONCURRENT}"
  SUBMIT=(
    --account="${SEAL_GPU_ACCOUNT:-pilot_sae_gpu}"
    --partition="${SEAL_GPU_PARTITION:-sae}"
    --constraint="${SEAL_GPU_CONSTRAINT:-ampere}"
    --time="${SEAL_GPU_TIME:-24:00:00}"
    --job-name="seal-weighted-${CONFIG_EXPERIMENTS}x${CONFIG_FOODWEBS}x4"
  )
fi
SUBMIT+=(--parsable --export=ALL --array="${ARRAY_SPEC}" --output="${LOG_PATTERN}")
if [[ "${SUBMIT_HELD:-1}" == "1" ]]; then SUBMIT+=(--hold); fi
JOB_ID="$(sbatch "${SUBMIT[@]}" "${SCRIPT}")"
printf '%s\n' "${JOB_ID}" > "${RUN_ROOT}/slurm/${MODE}_job_id.txt"
{
  printf 'mode=%s\n' "${MODE}"
  printf 'array=%s\n' "${ARRAY_SPEC}"
  printf 'experiments=%s\n' "${CONFIG_EXPERIMENTS}"
  printf 'foodwebs=%s\n' "${CONFIG_FOODWEBS}"
  printf 'variants=4\n'
  printf 'max_concurrent=%s\n' "${MAX_CONCURRENT}"
} > "${RUN_ROOT}/slurm/${MODE}_submission_config.txt"
echo "[INFO] Submitted ${MODE} job ${JOB_ID}"
echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] Inspect the held array before release."
