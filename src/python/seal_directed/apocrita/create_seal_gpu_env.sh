#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[ERROR] Create the environment inside an salloc/sbatch allocation, not on a login node." >&2
  exit 2
fi
if [[ -z "${SEAL_ENV_PREFIX:-}" ]]; then
  echo "[ERROR] Set SEAL_ENV_PREFIX to persistent project storage." >&2
  exit 2
fi
: "${SCRATCH_ROOT:?Set SCRATCH_ROOT to /gpfs/scratch/$USER/... for disposable caches}"

mkdir -p "${SCRATCH_ROOT}/conda-pkgs" "${SCRATCH_ROOT}/pip-cache"
export CONDA_PKGS_DIRS="${SCRATCH_ROOT}/conda-pkgs"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/pip-cache"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DGCNN_LIB="$(cd "${SCRIPT_DIR}/../../pytorch_DGCNN/lib" && pwd)"

module load miniforge
mamba env create --prefix "${SEAL_ENV_PREFIX}" --file "${SCRIPT_DIR}/environment.apocrita.yml"
"${SEAL_ENV_PREFIX}/bin/python" -m pip install \
  torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${SEAL_ENV_PREFIX}/bin/python" -m pip install \
  --requirement "${SCRIPT_DIR}/requirements-apocrita.txt"

make -C "${DGCNN_LIB}" clean
make -C "${DGCNN_LIB}"
file "${DGCNN_LIB}/build/dll/libgnn.so"
"${SEAL_ENV_PREFIX}/bin/python" -m pip freeze > "${SEAL_ENV_PREFIX}/pip-freeze.txt"
conda env export --prefix "${SEAL_ENV_PREFIX}" > "${SEAL_ENV_PREFIX}/environment-export.yml"

"${SEAL_ENV_PREFIX}/bin/python" - <<'PY'
import json
import torch

payload = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, indent=2, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the new environment")
PY
