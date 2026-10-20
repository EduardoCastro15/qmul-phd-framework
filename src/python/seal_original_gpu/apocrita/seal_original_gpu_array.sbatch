#!/usr/bin/env bash
#SBATCH --job-name=seal-original
#SBATCH --output=seal-original.o%A.%a
#SBATCH --partition=gpu
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%2

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the deployed bundle}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX to the persistent Miniforge environment}"
: "${MANIFEST_PATH:?Set MANIFEST_PATH to manifest.csv}"
: "${CONFIG_PATH:?Set CONFIG_PATH to RUN_CONFIG.json}"
: "${PROJECT_RESULTS_ROOT:?Set PROJECT_RESULTS_ROOT to the persistent run root}"
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT to /gpfs/scratch/$USER/...}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT to the verified deployment commit}"
: "${TMPDIR:?Slurm did not provide TMPDIR}"

SEAL_DIR="${REPO_ROOT}/src/python/seal_original_gpu"
PYTHON_BIN="${SEAL_ENV_PREFIX}/bin/python"

module load miniforge
mkdir -p "${SCRATCH_ROOT}/torch-cache" "${SCRATCH_ROOT}/xdg-cache"
export TORCH_HOME="${SCRATCH_ROOT}/torch-cache"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"

mkdir -p "${PROJECT_RESULTS_ROOT}/gpu_monitor"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 60 \
  > "${PROJECT_RESULTS_ROOT}/gpu_monitor/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

"${PYTHON_BIN}" "${SEAL_DIR}/apocrita/gpu_probe.py"
"${PYTHON_BIN}" "${SEAL_DIR}/run_foodweb_seal_original_gpu.py" \
  --manifest "${MANIFEST_PATH}" \
  --config "${CONFIG_PATH}" \
  --result-root "${PROJECT_RESULTS_ROOT}" \
  --python-executable "${PYTHON_BIN}" \
  --device cuda \
  --require-cuda \
  --scratch-root "${TMPDIR}" \
  --source-commit "${SOURCE_COMMIT}" \
  --resume
