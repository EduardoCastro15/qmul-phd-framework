# SEAL Directed

This folder contains the directed SEAL pipeline. It is intentionally separate
from the existing SEAL runner in `src/python`.

## What Is Directed

- Positive links are ordered pairs from `net[source, target]`.
- Test masking removes only `source -> target`, not `target -> source`.
- Negative links are ordered absent pairs where `source != target`.
- By default, negative sampling uses ecological role filtering:
  source nodes must be resources or consumer-resources, and target nodes must
  be consumers or consumer-resources.
- Source and target nodes receive distinct labels in the enclosing subgraph.

## Current Model Limitation

The bundled DGCNN backend builds undirected message-passing matrices. This
pipeline therefore uses directed candidates and directed masking, then trains
on the weak projection of each enclosing subgraph while preserving pair order
through source/target labels.

## Node Attributes

Node attributes are integrated only in this directed pipeline. The attributed
MAT files live in:

```text
src/python/seal_directed/data/foodwebs_mat_seal_attrs/
```

They contain the original `net`, `taxonomy`, `mass`, and `role`, plus `group`,
the node feature matrix read by `--use-attribute`.

Regenerate them with:

```bash
/Users/acw792/miniconda3/envs/Foodweb/bin/python \
  src/python/seal_directed/build_node_attribute_mats.py \
  --overwrite
```

## Frozen 100 x 290 GPU protocol

The Apocrita campaign is a new cohort of 100 independent runs for every one of
the 290 food webs. It deliberately preserves the SEAL-directed configuration
rather than copying the WLNM protocol:

- independent 90/10 positive splits;
- one sampled negative per positive in train and test;
- the current role-compatible negative pool, with full-pool fallback when the
  constrained pool is too small;
- all 44 node attributes, no node2vec embedding;
- hop 1, batch size 50, 50 epochs and a fixed `score > 0.5` threshold;
- raw aggregation only (no Tukey filtering).

Seeds are unique and reproducible across all 29,000 runs:

```text
seed = 12345 + 100 * (foodweb_index - 1) + experiment_id - 1
```

Because SEAL and WLNM retain their own negative sampling, class balance and
input features, their outputs are comparisons of complete configurations. In
particular, PR-AUC, Precision and F1 should not be interpreted as isolating the
model architecture alone.

## Manifest-driven local example

```bash
RUN_ROOT="$(mktemp -d)"
PYTHON=/Users/acw792/miniconda3/envs/Foodweb/bin/python

"${PYTHON}" src/python/seal_directed/generate_seal_manifest.py \
  --foodweb-csv src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv \
  --mat-folder src/python/seal_directed/data/foodwebs_mat_seal_attrs \
  --output-dir "${RUN_ROOT}/manifest" \
  --num-experiments 1 \
  --expected-foodwebs 1 \
  --only-foodweb AEW01_tax_mass

"${PYTHON}" src/python/seal_directed/run_foodweb_seal_directed.py \
  --manifest "${RUN_ROOT}/manifest/manifest.csv" \
  --config "${RUN_ROOT}/manifest/RUN_CONFIG.json" \
  --result-root "${RUN_ROOT}" \
  --experiment-id 1 \
  --device cpu \
  --num-workers 1
```

Each successful run is an immutable directory containing `metrics.json`, a
compressed `artifacts.npz` and a checksummed `_SUCCESS` marker. Reruns with
`--resume` skip only bundles that pass validation with the same configuration
hash. Invalid or mismatched output fails closed.

Validate and create the raw summaries with:

```bash
"${PYTHON}" src/python/seal_directed/validate_seal_directed_gpu_run.py \
  --manifest "${RUN_ROOT}/manifest/manifest.csv" \
  --config "${RUN_ROOT}/manifest/RUN_CONFIG.json" \
  --result-root "${RUN_ROOT}" \
  --write-summaries
```

## Apocrita GPU workflow

The scripts under `apocrita/` implement the guarded workflow:

1. Allocate a compute node and create the Miniforge environment with
   `create_seal_gpu_env.sh`. This also recompiles the DGCNN library for Linux.
2. Export `REPO_ROOT`, `SEAL_ENV_PREFIX`, `PROJECT_RESULTS_ROOT`, `SCRATCH_ROOT`
   and an explicit smoke `RUN_ID`.
3. Submit `submit_seal_directed_gpu.sh smoke`. Submissions are held by default;
   inspect and release the job deliberately. The smoke run has its own 3 x 1
   manifest and must not share a `RUN_ID` with production.
4. Use the smoke job ID and GPU monitor CSV to request production GPU access.
5. Choose a new production `RUN_ID` and submit `calibration` for ExperimentID 1.
6. After setting the measured walltime in `seal_directed_gpu_array.sbatch`,
   submit `full` with that same production `RUN_ID`; this schedules
   ExperimentID 2-100.
7. Run the validator with the Slurm job IDs before declaring completion.

For example, environment creation starts from an interactive allocation:

```bash
salloc -p gpushort -n 8 --cpus-per-gpu=8 --mem-per-cpu=11G \
  --gres=gpu:1 -t 1:0:0
module load miniforge
export SEAL_ENV_PREFIX=/path/to/project/envs/seal-directed-gpu
export SCRATCH_ROOT=/gpfs/scratch/$USER/seal-directed
bash src/python/seal_directed/apocrita/create_seal_gpu_env.sh
```

The production validation must include CUDA and both the calibration and main
array job IDs:

```bash
"${SEAL_ENV_PREFIX}/bin/python" \
  src/python/seal_directed/validate_seal_directed_gpu_run.py \
  --manifest "${RUN_ROOT}/manifest/manifest.csv" \
  --config "${RUN_ROOT}/manifest/RUN_CONFIG.json" \
  --result-root "${RUN_ROOT}" \
  --job-id "${CALIBRATION_JOB_ID}" \
  --job-id "${FULL_ARRAY_JOB_ID}" \
  --require-cuda \
  --write-summaries
```

Required persistent variables:

```text
PROJECT_RESULTS_ROOT  persistent, backed-up project storage
SEAL_ENV_PREFIX       persistent Miniforge prefix
REPO_ROOT             clean deployed repository checkout
SCRATCH_ROOT           disposable staging and package/model caches under /gpfs/scratch
```

The full array requests one GPU per task and starts at two simultaneous tasks.
CUDA is mandatory and the job aborts rather than falling back to CPU. Never set
`CUDA_VISIBLE_DEVICES` manually; Slurm owns that mapping.

Each task stages the current MAT input under Slurm's `$TMPDIR`; final atomic run
bundles, Slurm logs, job IDs, source snapshot and environment manifests remain
under `PROJECT_RESULTS_ROOT`.

Deterministic algorithms are frozen in `RUN_CONFIG.json`. If the smoke test
identifies a specific unsupported CUDA operation, start a distinct `RUN_ID`
with `SEAL_DETERMINISTIC=0`; deterministic and non-deterministic outputs cannot
be mixed by resume because they have different configuration hashes.
