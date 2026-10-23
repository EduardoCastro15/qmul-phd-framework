#!/usr/bin/env bash
#SBATCH --job-name=seal-weighted
#SBATCH --partition=sae
#SBATCH --ntasks=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-40%2

set -euo pipefail
: "${REPO_ROOT:?Set REPO_ROOT}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX}"
: "${MANIFEST_PATH:?Set MANIFEST_PATH}"
: "${CONFIG_PATH:?Set CONFIG_PATH}"
: "${PROJECT_RESULTS_ROOT:?Set PROJECT_RESULTS_ROOT}"
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT}"
: "${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
: "${TMPDIR:?Slurm did not provide TMPDIR}"

WEIGHTED_DIR="${REPO_ROOT}/src/python/seal_directed_weighted"
PYTHON_BIN="${SEAL_ENV_PREFIX}/bin/python"
module load miniforge
mkdir -p "${SCRATCH_ROOT}/torch-cache" "${SCRATCH_ROOT}/xdg-cache"
export TORCH_HOME="${SCRATCH_ROOT}/torch-cache"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${PROJECT_RESULTS_ROOT}/gpu_monitor"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 60 \
  > "${PROJECT_RESULTS_ROOT}/gpu_monitor/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

"${PYTHON_BIN}" "${WEIGHTED_DIR}/apocrita/gpu_probe.py"
"${PYTHON_BIN}" "${WEIGHTED_DIR}/run_campaign.py" \
  --manifest "${MANIFEST_PATH}" \
  --config "${CONFIG_PATH}" \
  --result-root "${PROJECT_RESULTS_ROOT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --python-executable "${PYTHON_BIN}" \
  --device cuda \
  --require-cuda \
  --num-workers "${SEAL_NUM_WORKERS:-4}" \
  --scratch-root "${TMPDIR}" \
  --source-commit "${SOURCE_COMMIT}" \
  --resume

