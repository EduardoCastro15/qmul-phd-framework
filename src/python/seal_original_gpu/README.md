# SEAL original on CUDA

This directory contains only the operational layer required to execute the
historical `src/python/Main.py` implementation on CUDA and Apocrita. It does
not replace or correct the original sampler, graph construction, training,
metrics, or seeds.

## Frozen campaign

- 290 original MAT files from `src/matlab/data/foodwebs_mat`.
- 10 experiments with seeds `12345..12354`.
- One experiment seed is deliberately reused for all 290 food webs, matching
  `run_foodweb_seal.py`.
- Requested train ratio 0.9, hop 1, batch size 50, 100000 maximum training
  samples, 50 epochs, and threshold 0.5.
- Parallel subgraph extraction remains enabled.
- Attributes, embeddings, and `all_unknown_as_negative` remain disabled.
- Raw results only; no Tukey filtering.

`Main.py`, `util_functions.py`, and `run_foodweb_seal.py` are hashed into every
`RUN_CONFIG.json`. A run refuses to start if any of those files differs from
the frozen hashes.

## Local checks

```bash
cd /Users/acw792/Developer/qmul-phd-framework

env PYTHONDONTWRITEBYTECODE=1 \
  /Users/acw792/miniconda3/envs/Foodweb/bin/python \
  -m unittest discover \
  -s src/python/seal_original_gpu/tests \
  -p 'test_*.py'
```

Generate a dry-run manifest without executing training:

```bash
TMP_ROOT="$(mktemp -d /private/tmp/seal-original-manifest-XXXXXX)"

/Users/acw792/miniconda3/envs/Foodweb/bin/python \
  src/python/seal_original_gpu/generate_seal_original_manifest.py \
  --repo-root "$PWD" \
  --foodweb-csv src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv \
  --mat-folder src/matlab/data/foodwebs_mat \
  --output-dir "${TMP_ROOT}" \
  --num-experiments 10 \
  --expected-foodwebs 290 \
  --source-commit "$(git rev-parse HEAD)"
```

## Bundle and Apocrita

The bundle builder requires all included inputs to be committed:

```bash
BUNDLE_OUT="$(mktemp -d /private/tmp/seal-original-upload-XXXXXX)"

/Users/acw792/miniconda3/envs/Foodweb/bin/python \
  src/python/seal_original_gpu/apocrita/build_seal_original_apocrita_bundle.py \
  --output-dir "${BUNDLE_OUT}"
```

Upload the resulting `.tar.gz` and `.sha256` files, verify them with
`sha256sum -c`, and extract the archive under a dedicated directory such as:

```bash
$HOME/PhD/Stage2/Code/seal-original-gpu
```

On Apocrita, create a dedicated Python 3.12 / PyTorch 2.11.0+cu126 environment
inside a CPU allocation. `libgnn.so` is compiled with `/usr/bin/g++` to avoid
the Conda `GLIBCXX` mismatch.

```bash
export DEPLOYED_ROOT="$HOME/PhD/Stage2/Code/seal-original-gpu/seal-original-apocrita-<commit>"
export SCRATCH_ROOT="/gpfs/scratch/$USER/SEAL_original"
export SEAL_ENV_PREFIX="${SCRATCH_ROOT}/envs/seal-original-py312-cu126"

salloc --partition=compute --ntasks=1 --mem-per-cpu=8G --time=02:00:00

bash \
  "${DEPLOYED_ROOT}/src/python/seal_original_gpu/apocrita/create_seal_original_gpu_env.sh"
```

After leaving the allocation, submit a held smoke test:

```bash
export REPO_ROOT="${DEPLOYED_ROOT}"
export PROJECT_RESULTS_ROOT="$HOME/PhD/Stage2/Results/SEAL_original/smoke"
export RUN_ID="seal_original_smoke_<commit>_<date>"
export SUBMIT_HELD=1

bash \
  "${DEPLOYED_ROOT}/src/python/seal_original_gpu/apocrita/submit_seal_original_gpu.sh" \
  smoke
```

Submissions support four modes:

- `smoke`: three food webs on `gpushort`.
- `calibration`: ExperimentID 1 over all 290 food webs.
- `full`: ExperimentID 2-10 after a successful calibration with the same
  `RUN_ID`.
- `campaign`: ExperimentID 1-10 in a single held array.

Every submission is held by default. Inspect the manifest, Slurm resources,
and output path before releasing it.

The final validator is:

```bash
"${SEAL_ENV_PREFIX}/bin/python" \
  "${DEPLOYED_ROOT}/src/python/seal_original_gpu/validate_seal_original_gpu_run.py" \
  --manifest "${RUN_ROOT}/manifest/manifest.csv" \
  --config "${RUN_ROOT}/manifest/RUN_CONFIG.json" \
  --result-root "${RUN_ROOT}" \
  --job-id "${CALIBRATION_JOB_ID}" \
  --job-id "${FULL_JOB_ID}" \
  --require-cuda \
  --write-summaries
```

The validator expects 2,900 completed runs but exactly 10 distinct numeric
seeds. It intentionally does not reject duplicate negatives, inverse-edge
negatives, or other behavior inherited from the original sampler.
