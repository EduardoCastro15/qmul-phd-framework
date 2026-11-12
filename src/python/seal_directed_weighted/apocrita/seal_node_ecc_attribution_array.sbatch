#!/usr/bin/env bash
#SBATCH --job-name=seal-node-ecc-v2
#SBATCH --partition=sae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=88G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-20%2

set -euo pipefail
: "${REPO_ROOT:?Set REPO_ROOT}"
: "${SEAL_ENV_PREFIX:?Set SEAL_ENV_PREFIX}"
: "${MANIFEST_PATH:?Set MANIFEST_PATH}"
: "${RUN_CONFIG_PATH:?Set RUN_CONFIG_PATH}"
: "${TRAINING_RESULT_ROOT:?Set TRAINING_RESULT_ROOT}"
: "${PIPELINE_STAGE:?Set PIPELINE_STAGE to training or attribution}"
: "${TMPDIR:?Slurm did not provide TMPDIR}"

WEIGHTED_DIR="${REPO_ROOT}/src/python/seal_directed_weighted"
PYTHON_BIN="${SEAL_ENV_PREFIX}/bin/python"
module load miniforge
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_HOME="${SCRATCH_ROOT:-${TMPDIR}}/torch-cache"
export XDG_CACHE_HOME="${SCRATCH_ROOT:-${TMPDIR}}/xdg-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT:-${TMPDIR}}/mpl-cache"
mkdir -p "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"
MONITOR_ROOT="${ATTRIBUTION_RESULT_ROOT:-${TRAINING_RESULT_ROOT}}"
mkdir -p "${MONITOR_ROOT}/gpu_monitor"
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 60 \
  > "${MONITOR_ROOT}/gpu_monitor/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${WEIGHTED_DIR}/apocrita/gpu_probe.py"

COMMON=(
  --manifest "${MANIFEST_PATH}"
  --result-root "${TRAINING_RESULT_ROOT}"
  --python-executable "${PYTHON_BIN}"
  --device cuda --require-cuda --resume
)
if [[ "${TASK_SELECTION:-shard}" == "block" ]]; then
  COMMON+=(
    --block-task-id "${SLURM_ARRAY_TASK_ID}"
    --foodweb-block-size "${FOODWEB_BLOCK_SIZE:-6}"
  )
elif [[ "${TASK_SELECTION:-shard}" == "row" ]]; then
  COMMON+=(--manifest-row-id "${SLURM_ARRAY_TASK_ID}")
else
  COMMON+=(--task-id "${SLURM_ARRAY_TASK_ID}")
fi
if [[ -n "${ONLY_FOODWEB:-}" ]]; then COMMON+=(--only-foodweb "${ONLY_FOODWEB}"); fi

if [[ "${PIPELINE_STAGE}" == "training" ]]; then
  "${PYTHON_BIN}" "${WEIGHTED_DIR}/run_campaign.py" \
    "${COMMON[@]}" --config "${RUN_CONFIG_PATH}" \
    --scratch-root "${TMPDIR}" --source-commit "${SOURCE_COMMIT:-unknown}" \
    --num-workers "${SEAL_NUM_WORKERS:-4}"
elif [[ "${PIPELINE_STAGE}" == "attribution" ]]; then
  : "${ATTRIBUTION_CONFIG_PATH:?Set ATTRIBUTION_CONFIG_PATH}"
  : "${ATTRIBUTION_RESULT_ROOT:?Set ATTRIBUTION_RESULT_ROOT}"
  COMMAND=(
    "${PYTHON_BIN}" "${WEIGHTED_DIR}/run_attribution_campaign.py"
    --manifest "${MANIFEST_PATH}" --run-config "${RUN_CONFIG_PATH}"
    --attribution-config "${ATTRIBUTION_CONFIG_PATH}"
    --training-result-root "${TRAINING_RESULT_ROOT}"
    --result-root "${ATTRIBUTION_RESULT_ROOT}"
    --python-executable "${PYTHON_BIN}"
    --device cuda --require-cuda --resume
  )
  if [[ "${TASK_SELECTION:-shard}" == "block" ]]; then
    COMMAND+=(
      --block-task-id "${SLURM_ARRAY_TASK_ID}"
      --foodweb-block-size "${FOODWEB_BLOCK_SIZE:-6}"
    )
  elif [[ "${TASK_SELECTION:-shard}" == "row" ]]; then
    COMMAND+=(--manifest-row-id "${SLURM_ARRAY_TASK_ID}")
  else
    COMMAND+=(--task-id "${SLURM_ARRAY_TASK_ID}")
  fi
  if [[ -n "${HISTORICAL_RUN_ROOT:-}" ]]; then
    COMMAND+=(--historical-result-root "${HISTORICAL_RUN_ROOT}")
  fi
  if [[ -n "${ONLY_FOODWEB:-}" ]]; then COMMAND+=(--only-foodweb "${ONLY_FOODWEB}"); fi
  "${COMMAND[@]}"
else
  echo "Unknown PIPELINE_STAGE=${PIPELINE_STAGE}" >&2; exit 2
fi
