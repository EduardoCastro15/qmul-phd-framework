#!/usr/bin/env bash
#SBATCH --job-name=seal-nodeattr-attr
#SBATCH --output=seal-nodeattr-attr.o%A.%a
#SBATCH --partition=gpu
#SBATCH --ntasks=4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=11G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%2

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the deployed qmul-phd-framework checkout}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX to the persistent Miniforge environment}"
: "${MANIFEST_PATH:?Set MANIFEST_PATH to manifest.csv}"
: "${CONFIG_PATH:?Set CONFIG_PATH to RUN_CONFIG.json}"
: "${ATTRIBUTION_CONFIG_PATH:?Set ATTRIBUTION_CONFIG_PATH}"
: "${TRAINING_RESULTS_ROOT:?Set TRAINING_RESULTS_ROOT}"
: "${ATTRIBUTION_RESULTS_ROOT:?Set ATTRIBUTION_RESULTS_ROOT}"
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT to /gpfs/scratch/$USER/...}"
: "${TMPDIR:?Slurm did not provide TMPDIR}"

SEAL_DIR="${REPO_ROOT}/src/python/seal_directed"
PYTHON_BIN="${SEAL_ENV_PREFIX}/bin/python"

module load miniforge
mkdir -p "${SCRATCH_ROOT}/torch-cache" "${SCRATCH_ROOT}/xdg-cache" "${SCRATCH_ROOT}/mpl-cache"
export TORCH_HOME="${SCRATCH_ROOT}/torch-cache"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/mpl-cache"

mkdir -p "${ATTRIBUTION_RESULTS_ROOT}/attributions/gpu_monitor"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 60 \
  > "${ATTRIBUTION_RESULTS_ROOT}/attributions/gpu_monitor/job_${SLURM_ARRAY_JOB_ID}.csv" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

"${PYTHON_BIN}" "${SEAL_DIR}/apocrita/gpu_probe.py"
COMMAND=(
  "${PYTHON_BIN}" "${SEAL_DIR}/run_foodweb_seal_attribution.py"
  --manifest "${MANIFEST_PATH}"
  --run-config "${CONFIG_PATH}"
  --attribution-config "${ATTRIBUTION_CONFIG_PATH}"
  --training-result-root "${TRAINING_RESULTS_ROOT}"
  --result-root "${ATTRIBUTION_RESULTS_ROOT}"
  --python-executable "${PYTHON_BIN}"
  --device cuda
  --require-cuda
  --scratch-root "${TMPDIR}"
  --resume
)
if [[ -n "${SANITY_RANDOMIZE_THROUGH:-}" ]]; then
  COMMAND+=(--randomize-through "${SANITY_RANDOMIZE_THROUGH}")
fi
"${COMMAND[@]}"
