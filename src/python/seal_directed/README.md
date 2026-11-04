# SEAL-D-NodeAttr

Scientific display name: **SEAL-D-NodeAttr**.

Stable identifier for new campaigns: `SEAL_directed_nodeattr_v1`.

Historical alias retained for old results and paths: `SEAL_directed`

No historical folder, campaign or result is renamed. New bundles add
`ModelID`, `DisplayName` and `LegacyFamily`; `Version=SEAL_directed` remains as
a compatibility field.

This folder contains the directed, node-attributed SEAL pipeline and its
post-training attribution workflow. It is intentionally separate from the
existing SEAL runner in `src/python`.

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

## Frozen 10 x 290 GPU protocol

The Apocrita campaign is a new cohort of 10 independent runs for every one of
the 290 food webs. It deliberately preserves the SEAL-directed configuration
rather than copying the WLNM protocol:

- independent 90/10 positive splits;
- one sampled negative per positive in train and test;
- the current role-compatible negative pool, with full-pool fallback when the
  constrained pool is too small;
- all 44 node attributes, no node2vec embedding;
- hop 1, batch size 50, 50 epochs and a fixed `score > 0.5` threshold;
- raw aggregation only (no Tukey filtering).

Seeds are unique and reproducible across all 2,900 runs:

```text
seed = 12345 + 10 * (foodweb_index - 1) + experiment_id - 1
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

Each successful new run is an immutable version-2 directory containing
`metrics.json`, `artifacts.npz`, `checkpoint.pt`, `model_config.json` and a
checksummed `_SUCCESS` marker. The artifacts preserve exact ragged train/test
subgraphs (candidate pairs, node IDs, DRNL labels and local edge order) so an
explanation never reruns stochastic subgraph extraction. Reruns with
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
4. Use the smoke job ID and GPU monitor CSV to confirm production GPU access.
5. For a new deployment, choose a production `RUN_ID`, submit `calibration`
   for ExperimentID 1, inspect it, and then submit `full` with that same
   `RUN_ID` for ExperimentID 2-10.
6. When an equivalent deployment has already passed calibration, choose a new
   production `RUN_ID` and submit `campaign` to schedule ExperimentID 1-10 in
   one array. Set `SEAL_NUM_EXPERIMENTS=10` explicitly whenever creating or
   resuming the run.
7. Run the validator with every production Slurm job ID before declaring
   completion.

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

## Attribute attribution

The explainer is a separate, retryable process. It holds topology fixed and
targets positive log odds,
`log(p_positive) - log(p_negative)`, rather than softmax probability.

- The 44 ecological channels use Expected Gradients with empirical baselines
  drawn only from frozen training subgraphs. Baseline row 0 is a training
  resource/source, row 1 is a training consumer/target, and rows 2 onward are
  contexts from that same training subgraph. If the reference context is too
  short, all of it is retained and only missing rows are sampled with
  replacement from that context.
- DRNL one-hot inputs use Integrated Gradients from an all-zero baseline while
  real attributes and topology remain fixed.
- DRNL and ecological-attribute values are never summed or compared as one
  scale. They answer different model-reliance questions.
- Edge attribution is out of scope because the current sparse message-passing
  topology is not represented as differentiable edge weights.

Captum is frozen at `0.9.0`. The inherited DGCNN no longer re-wraps input
tensors in the obsolete `Variable` API, because that operation detached node
inputs under modern PyTorch. The legacy `forward(batch_graph)` and the explicit
differentiable forward are tested for prediction parity at `1e-7`.

Create a conservative configuration for smoke/calibration:

```bash
"${PYTHON}" src/python/seal_directed/generate_attribution_config.py \
  --output "${RUN_ROOT}/calibration/config_256/ATTRIBUTION_CONFIG.json" \
  --expected-gradients-samples 256 \
  --drnl-steps 256
```

Explain one validated training bundle:

```bash
"${PYTHON}" src/python/seal_directed/explain_seal_directed.py \
  --training-run-dir "${RUN_ROOT}/runs/AEW01_tax_mass/experiment_001_seed_12345" \
  --mat-path src/python/seal_directed/data/foodwebs_mat_seal_attrs/AEW01_tax_mass.mat \
  --attribution-config "${RUN_ROOT}/manifest/ATTRIBUTION_CONFIG.json" \
  --output-dir "${RUN_ROOT}/attributions/runs/AEW01_tax_mass/experiment_001_seed_12345" \
  --device cpu
```

Every attribution bundle contains candidate pairs, labels, frozen scores,
TP/TN/FP/FN codes, ragged node offsets/IDs/roles, a `total_nodes x 44`
Expected-Gradients matrix, within-candidate normalized absolute attribution,
DRNL labels and IG matrices, background training indices, completeness errors,
software versions and hashes of the training `_SUCCESS`, checkpoint and feature
schema.

## Calibration and release gate

For ExperimentID 1, generate four immutable configs and attribution roots for
32, 64, 128 and 256 samples/steps. Then freeze the least-cost pair that meets
all gates relative to 256:

```bash
"${PYTHON}" src/python/seal_directed/freeze_attribution_config.py \
  --manifest "${RUN_ROOT}/manifest/manifest.csv" \
  --calibration-root "32=${RUN_ROOT}/calibration/results_32" \
  --calibration-root "64=${RUN_ROOT}/calibration/results_64" \
  --calibration-root "128=${RUN_ROOT}/calibration/results_128" \
  --calibration-root "256=${RUN_ROOT}/calibration/results_256" \
  --output-dir "${RUN_ROOT}/manifest"
```

The automatic gate requires median Spearman >= 0.95, median top-10 overlap >=
0.80, median relative completeness error <= 1%, and 95th-percentile relative
completeness error <= 5%. Failure writes a failed calibration report and does
not create a final `ATTRIBUTION_CONFIG.json`; the Apocrita submission wrapper
then refuses to release ExperimentID 2-10.

The guarded sequence is:

1. train and explain a 3-web x 1 smoke cohort;
2. train ExperimentID 1 for 290 webs;
3. run four calibration attribution roots and freeze the config;
4. write definitive ExperimentID 1 attributions with that config;
5. train ExperimentID 2-10;
6. run the attribution array for ExperimentID 2-10;
7. validate `2900/2900` training and `2900/2900` attribution bundles with
   `validate_attribution_campaign.py`, including all `sacct` job IDs.

`apocrita/submit_nodeattr_attribution.sh` submits these stages held by default.
It refuses campaign mode without `CALIBRATION_REPORT.json` containing
`CALIBRATION_OK`.

## Sanity checks, fidelity and summaries

`explain_seal_directed.py --randomize-through` supports progressive cumulative
reinitialization of the output, MLP, SortPooling head and message-passing
layers. A separate `RUN_CONFIG` with `label_permutation_control=true` trains the
permuted-label control. For the 3-web smoke cohort,
`validate_attribution_sanity.py` requires median absolute attribution
correlation below 0.5 for the fully randomized and permuted-label controls.
Create that distinct control manifest with the same three smoke webs and
`generate_seal_manifest.py --label-permutation-control`; never resume it into
the valid-model result root.

For the faithfulness-only check, create a sanity attribution config with
`--faithfulness-background-samples 32`. Top- and bottom-ranked feature families
are replaced by empirical training backgrounds; the validator requires a
larger median positive-log-odds reduction for top-ranked families. This
substitution is a diagnostic, not the attribution method.

`summarize_attributions.py` aggregates candidate -> run -> food web -> cohort,
gives every food web equal global weight, and bootstraps food webs 10,000
times. It writes separate outputs for 44 channels, six feature families,
resource/consumer/context, confusion class, ecosystem, negative-pool fallback,
and DRNL. Normalized absolute attribution is primary and signed attribution is
secondary. No exploratory p-values are produced, and all interpretation is
explicitly “model reliance on input attributes,” not a causal effect.

## Methodological references

- Zhang, M. & Chen, Y. (2018). Link Prediction Based on Graph Neural Networks.
  *NeurIPS 31* (SEAL).
- Zhang, M. et al. (2018). An End-to-End Deep Learning Architecture for Graph
  Classification. *AAAI 32* (DGCNN), doi:10.1609/aaai.v32i1.11782.
- Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic Attribution for Deep
  Networks. *ICML*, PMLR 70 (Integrated Gradients).
- Erion, G. et al. (2021). Improving performance of deep learning models with
  axiomatic attribution priors and expected gradients. *Nature Machine
  Intelligence* 3, 620-631, doi:10.1038/s42256-020-00265-3.
- Adebayo, J. et al. (2018). Sanity Checks for Saliency Maps. *NeurIPS 31*.
- Yeh, C.-K. et al. (2019). On the (In)fidelity and Sensitivity of
  Explanations. *NeurIPS 32*.
- Sanchez-Lengeling, B. et al. (2020). Evaluating Attribution for Graph Neural
  Networks. *NeurIPS 33*.
