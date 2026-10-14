#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "calibration" && "${MODE}" != "full" && "${MODE}" != "campaign" ]]; then
  echo "Usage: $0 {smoke|calibration|full|campaign}" >&2
  exit 2
fi

: "${REPO_ROOT:?Set REPO_ROOT to a clean deployed checkout}"
: "${PROJECT_RESULTS_ROOT:?Set PROJECT_RESULTS_ROOT to persistent project storage}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX to the persistent Miniforge environment}"
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT to /gpfs/scratch/$USER/... for disposable staging and caches}"

REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
SEAL_DIR="${REPO_ROOT}/src/python/seal_directed"
PYTHON_BIN="${SEAL_ENV_PREFIX}/bin/python"
NATIVE_LIBRARY="${REPO_ROOT}/src/python/pytorch_DGCNN/lib/build/dll/libgnn.so"
RUN_ID="${RUN_ID:-seal_directed_gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${PROJECT_RESULTS_ROOT}/${RUN_ID}"
MANIFEST_DIR="${RUN_ROOT}/manifest"
REQUIRED_FREE_GB="${REQUIRED_FREE_GB:-20}"
SEAL_NUM_EXPERIMENTS="${SEAL_NUM_EXPERIMENTS:-10}"
SEAL_MAX_CONCURRENT="${SEAL_MAX_CONCURRENT:-2}"
DETERMINISM_MANIFEST_ARGS=()
if [[ "${SEAL_DETERMINISTIC:-1}" != "1" ]]; then
  DETERMINISM_MANIFEST_ARGS+=(--non-deterministic)
fi
if [[ "${MODE}" != "smoke" ]] && { [[ ! "${SEAL_NUM_EXPERIMENTS}" =~ ^[0-9]+$ ]] || (( SEAL_NUM_EXPERIMENTS < 2 )); }; then
  echo "[ERROR] SEAL_NUM_EXPERIMENTS must be an integer greater than or equal to 2." >&2
  exit 2
fi
if [[ ! "${SEAL_MAX_CONCURRENT}" =~ ^[0-9]+$ ]] || (( SEAL_MAX_CONCURRENT < 1 )); then
  echo "[ERROR] SEAL_MAX_CONCURRENT must be a positive integer." >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Missing environment Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${NATIVE_LIBRARY}" ]]; then
  echo "[ERROR] Missing native DGCNN library: ${NATIVE_LIBRARY}" >&2
  echo "[ERROR] Build it inside a compute allocation with: make -C ${REPO_ROOT}/src/python/pytorch_DGCNN/lib CXX=/usr/bin/g++" >&2
  exit 2
fi
DEPLOYMENT_MODE="bundle"
if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  DEPLOYMENT_MODE="git"
  SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    echo "[ERROR] Commit or deliberately snapshot the source changes before submission." >&2
    exit 2
  fi
else
  : "${SOURCE_COMMIT_FILE:=${REPO_ROOT}/SOURCE_COMMIT.txt}"
  if [[ ! -f "${SOURCE_COMMIT_FILE}" ]]; then
    echo "[ERROR] Bundle deployment is missing SOURCE_COMMIT.txt." >&2
    exit 2
  fi
  "${PYTHON_BIN}" "${SEAL_DIR}/apocrita/verify_deployment_bundle.py" --root "${REPO_ROOT}"
  SOURCE_COMMIT="$(tr -d '[:space:]' < "${SOURCE_COMMIT_FILE}")"
fi
if [[ ! -w "${PROJECT_RESULTS_ROOT}" ]]; then
  echo "[ERROR] PROJECT_RESULTS_ROOT is not writable: ${PROJECT_RESULTS_ROOT}" >&2
  exit 2
fi
mkdir -p "${SCRATCH_ROOT}"
if [[ ! -w "${SCRATCH_ROOT}" ]]; then
  echo "[ERROR] SCRATCH_ROOT is not writable: ${SCRATCH_ROOT}" >&2
  exit 2
fi

AVAILABLE_KB="$(df -Pk "${PROJECT_RESULTS_ROOT}" | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((REQUIRED_FREE_GB * 1024 * 1024))"
if (( AVAILABLE_KB < REQUIRED_KB )); then
  echo "[ERROR] Less than ${REQUIRED_FREE_GB} GiB is available under PROJECT_RESULTS_ROOT." >&2
  exit 2
fi
if command -v qmquota >/dev/null 2>&1; then
  qmquota -s
fi

if [[ "${MODE}" == "full" && ! -e "${RUN_ROOT}" ]]; then
  echo "[ERROR] Run calibration first with the same RUN_ID; refusing to skip ExperimentID 1." >&2
  exit 2
fi

if [[ ! -e "${RUN_ROOT}" ]]; then
  mkdir -p "${MANIFEST_DIR}"
  if [[ "${MODE}" == "smoke" ]]; then
    "${PYTHON_BIN}" "${SEAL_DIR}/generate_seal_manifest.py" \
      --foodweb-csv "${REPO_ROOT}/src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv" \
      --mat-folder "${SEAL_DIR}/data/foodwebs_mat_seal_attrs" \
      --output-dir "${MANIFEST_DIR}" \
      --num-experiments 1 \
      --expected-foodwebs 3 \
      --source-commit "${SOURCE_COMMIT}" \
      "${DETERMINISM_MANIFEST_ARGS[@]}" \
      --only-foodweb AEW01_tax_mass \
      --only-foodweb PP1I4_tax_mass \
      --only-foodweb 'Chesapeake Bay_tax_mass'
  else
    "${PYTHON_BIN}" "${SEAL_DIR}/generate_seal_manifest.py" \
      --foodweb-csv "${REPO_ROOT}/src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv" \
      --mat-folder "${SEAL_DIR}/data/foodwebs_mat_seal_attrs" \
      --output-dir "${MANIFEST_DIR}" \
      --num-experiments "${SEAL_NUM_EXPERIMENTS}" \
      --expected-foodwebs 290 \
      --source-commit "${SOURCE_COMMIT}" \
      "${DETERMINISM_MANIFEST_ARGS[@]}"
  fi
  if [[ "${DEPLOYMENT_MODE}" == "git" ]]; then
    git -C "${REPO_ROOT}" archive --format=tar.gz --output="${RUN_ROOT}/source_snapshot.tar.gz" HEAD
  else
    tar \
      --exclude='*/__pycache__' \
      --exclude='src/python/pytorch_DGCNN/lib/build' \
      -czf "${RUN_ROOT}/source_snapshot.tar.gz" \
      -C "${REPO_ROOT}" .
    cp "${REPO_ROOT}/BUNDLE_METADATA.json" "${RUN_ROOT}/"
    cp "${REPO_ROOT}/DEPLOYMENT_MANIFEST.sha256" "${RUN_ROOT}/"
  fi
  printf '%s\n' "${SOURCE_COMMIT}" > "${RUN_ROOT}/SOURCE_COMMIT.txt"
  mkdir -p "${RUN_ROOT}/environment" "${RUN_ROOT}/slurm"
  cp "${SEAL_DIR}/apocrita/environment.apocrita.yml" "${RUN_ROOT}/environment/"
  cp "${SEAL_DIR}/apocrita/requirements-apocrita.txt" "${RUN_ROOT}/environment/"
  if [[ -f "${SEAL_ENV_PREFIX}/pip-freeze.txt" ]]; then
    cp "${SEAL_ENV_PREFIX}/pip-freeze.txt" "${RUN_ROOT}/environment/"
  fi
  if [[ -f "${SEAL_ENV_PREFIX}/environment-export.yml" ]]; then
    cp "${SEAL_ENV_PREFIX}/environment-export.yml" "${RUN_ROOT}/environment/"
  fi
else
  if [[ ! -f "${MANIFEST_DIR}/manifest.csv" || ! -f "${MANIFEST_DIR}/RUN_CONFIG.json" ]]; then
    echo "[ERROR] Existing RUN_ROOT is incomplete: ${RUN_ROOT}" >&2
    exit 2
  fi
  FROZEN_SOURCE_COMMIT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "${MANIFEST_DIR}/RUN_CONFIG.json")"
  if [[ "${FROZEN_SOURCE_COMMIT}" != "${SOURCE_COMMIT}" ]]; then
    echo "[ERROR] RUN_CONFIG source commit ${FROZEN_SOURCE_COMMIT} does not match checkout ${SOURCE_COMMIT}." >&2
    exit 2
  fi
fi

mkdir -p "${RUN_ROOT}/slurm"

CONFIG_FOODWEBS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_foodwebs"])' "${MANIFEST_DIR}/RUN_CONFIG.json")"
CONFIG_EXPERIMENTS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_experiments"])' "${MANIFEST_DIR}/RUN_CONFIG.json")"
if [[ "${MODE}" == "smoke" ]]; then
  if [[ "${CONFIG_FOODWEBS}" != "3" || "${CONFIG_EXPERIMENTS}" != "1" ]]; then
    echo "[ERROR] Smoke RUN_ID does not contain the expected 3x1 manifest." >&2
    exit 2
  fi
elif [[ "${CONFIG_FOODWEBS}" != "290" || "${CONFIG_EXPERIMENTS}" != "${SEAL_NUM_EXPERIMENTS}" ]]; then
  echo "[ERROR] Production RUN_ID does not contain the expected 290x${SEAL_NUM_EXPERIMENTS} manifest." >&2
  exit 2
fi

export MANIFEST_PATH="${MANIFEST_DIR}/manifest.csv"
export CONFIG_PATH="${MANIFEST_DIR}/RUN_CONFIG.json"
export PROJECT_RESULTS_ROOT="${RUN_ROOT}"
export REPO_ROOT SEAL_ENV_PREFIX SCRATCH_ROOT SOURCE_COMMIT

if [[ "${MODE}" == "smoke" ]]; then
  SCRIPT="${SEAL_DIR}/apocrita/seal_directed_gpushort.sbatch"
  ARRAY_ARGS=()
  LOG_PATTERN="${RUN_ROOT}/slurm/%x.o%j"
elif [[ "${MODE}" == "calibration" ]]; then
  SCRIPT="${SEAL_DIR}/apocrita/seal_directed_gpu_array.sbatch"
  ARRAY_ARGS=(--array=1%1)
  LOG_PATTERN="${RUN_ROOT}/slurm/%x.o%A.%a"
elif [[ "${MODE}" == "campaign" ]]; then
  SCRIPT="${SEAL_DIR}/apocrita/seal_directed_gpu_array.sbatch"
  ARRAY_ARGS=(--array="1-${CONFIG_EXPERIMENTS}%${SEAL_MAX_CONCURRENT}")
  LOG_PATTERN="${RUN_ROOT}/slurm/%x.o%A.%a"
else
  SCRIPT="${SEAL_DIR}/apocrita/seal_directed_gpu_array.sbatch"
  ARRAY_ARGS=(--array="2-${CONFIG_EXPERIMENTS}%${SEAL_MAX_CONCURRENT}")
  LOG_PATTERN="${RUN_ROOT}/slurm/%x.o%A.%a"
fi

SUBMIT_ARGS=(--parsable --export=ALL --output="${LOG_PATTERN}")
if [[ "${MODE}" != "smoke" ]]; then
  SEAL_GPU_ACCOUNT="${SEAL_GPU_ACCOUNT:-pilot_sae_gpu}"
  SEAL_GPU_PARTITION="${SEAL_GPU_PARTITION:-sae}"
  SEAL_GPU_CONSTRAINT="${SEAL_GPU_CONSTRAINT:-ampere}"
  SEAL_GPU_TIME="${SEAL_GPU_TIME:-24:00:00}"
  SUBMIT_ARGS+=(
    --account="${SEAL_GPU_ACCOUNT}"
    --partition="${SEAL_GPU_PARTITION}"
    --time="${SEAL_GPU_TIME}"
    --job-name="seal-directed-${CONFIG_EXPERIMENTS}x${CONFIG_FOODWEBS}"
  )
  if [[ -n "${SEAL_GPU_CONSTRAINT}" ]]; then
    SUBMIT_ARGS+=(--constraint="${SEAL_GPU_CONSTRAINT}")
  fi
  {
    printf 'account=%s\n' "${SEAL_GPU_ACCOUNT}"
    printf 'partition=%s\n' "${SEAL_GPU_PARTITION}"
    printf 'constraint=%s\n' "${SEAL_GPU_CONSTRAINT}"
    printf 'time_limit=%s\n' "${SEAL_GPU_TIME}"
    printf 'num_experiments=%s\n' "${CONFIG_EXPERIMENTS}"
    printf 'max_concurrent=%s\n' "${SEAL_MAX_CONCURRENT}"
  } > "${RUN_ROOT}/slurm/${MODE}_submission_config.txt"
fi
if [[ "${SUBMIT_HELD:-1}" == "1" ]]; then
  SUBMIT_ARGS+=(--hold)
fi
JOB_ID="$(sbatch "${SUBMIT_ARGS[@]}" "${ARRAY_ARGS[@]}" "${SCRIPT}")"
printf '%s\n' "${JOB_ID}" > "${RUN_ROOT}/slurm/${MODE}_job_id.txt"
echo "[INFO] Submitted ${MODE} job ${JOB_ID}"
echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] Inspect the job and manifest before releasing a held submission."
