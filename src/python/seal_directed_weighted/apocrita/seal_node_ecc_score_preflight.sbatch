#!/usr/bin/env bash
#SBATCH --job-name=seal-v3-preflight
#SBATCH --partition=sae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=88G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
: "${REPO_ROOT:?Set REPO_ROOT}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${TRAINING_RESULT_ROOT:?Set TRAINING_RESULT_ROOT}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX}"
: "${ATTRIBUTION_SOURCE_COMMIT:?Set ATTRIBUTION_SOURCE_COMMIT}"

WEIGHTED_DIR="${REPO_ROOT}/src/python/seal_directed_weighted"
PYTHON_BIN="${SEAL_ENV_PREFIX}/bin/python"
MANIFEST_PATH="${RUN_ROOT}/manifest/manifest.csv"
RUN_CONFIG_PATH="${RUN_ROOT}/manifest/RUN_CONFIG.json"
ATTRIBUTION_CONFIG_PATH="${RUN_ROOT}/calibration/config_32/ATTRIBUTION_CONFIG.json"
OUTPUT_DIR="${RUN_ROOT}/preflight/score_replay"

module load miniforge
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR}/mpl-cache"
mkdir -p "${MPLCONFIGDIR}" "${RUN_ROOT}/gpu_monitor"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 60 \
  > "${RUN_ROOT}/gpu_monitor/preflight_${SLURM_JOB_ID}.csv" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

"${PYTHON_BIN}" "${WEIGHTED_DIR}/apocrita/gpu_probe.py"
"${PYTHON_BIN}" "${WEIGHTED_DIR}/preflight_score_replay.py" \
  --manifest "${MANIFEST_PATH}" \
  --run-config "${RUN_CONFIG_PATH}" \
  --attribution-config "${ATTRIBUTION_CONFIG_PATH}" \
  --training-result-root "${TRAINING_RESULT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-id 1 --expected-foodwebs 290 \
  --device cuda --require-cuda --deterministic \
  --attribution-source-commit "${ATTRIBUTION_SOURCE_COMMIT}"
