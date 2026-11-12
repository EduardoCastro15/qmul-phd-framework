# SEAL-D-NodeEdgeAttr

This package is an isolated extension of `src/python/seal_directed`. It keeps
the existing 44 node attributes and binary resource-to-consumer prediction,
then evaluates contextual edge covariates with truly directed message passing.
The directory name remains `seal_directed_weighted` for historical
reproducibility; the scientific alias changes without renaming old outputs.

## Scientific scope

Visible name: **SEAL-D-NodeEdgeAttr**

Future identifier: `SEAL_directed_nodeedgeattr_v1`

Historical alias retained in existing results: `SEAL_directed_weighted_v1`

The word *weighted* means **edge covariates**. It does not mean measured diet
proportion, biomass flux, interaction frequency, or interaction strength.
`full_v1` contains 17 channels:

- log10 consumer/resource mass ratio and two endpoint-mass missingness flags;
- eight multi-hot interaction-type channels (including `unknown`);
- three dimensionality channels (`2d`, `3d`, `unknown`);
- three classification channels (`ibi`, `nibi`, `unknown`).

Duplicate GATEWAy link records are aggregated deterministically. Taxon mass is
the within-web median of positive endpoint masses, while categorical conflicts
remain multi-hot. The target candidate never receives its own link metadata:
test positives and their attributes are removed from the observed network, and
every candidate edge is removed again from its enclosing subgraph.

The four frozen variants and their scientific display names are:

1. `legacy_weak_node` -> **SEAL-D-NodeAttr-WeakMP**: existing weak-projection DGCNN with node attributes;
2. `directed_node` -> **SEAL-D-NodeAttr-DirectedMP**: separate incoming/outgoing directed messages, no edge data;
3. `directed_scalar_gate` -> **SEAL-D-NodeEdgeAttr-Gate**: a learned 17 -> 32 -> 1 sigmoid gate;
4. `directed_ecc` -> **SEAL-D-NodeEdgeAttr-ECC**: layer- and direction-specific edge-conditioned filters.

This package also provides an opt-in, version-2 attribution workflow for
`directed_node` and `directed_ecc`.  It never rewrites the historical v1 run
bundles.  Both variants use Expected Gradients for the 44 node attributes and
Integrated Gradients for DRNL.  `directed_ecc` additionally uses Expected
Gradients for the 17 contextual edge covariates while keeping the local
directed topology fixed.  Node, DRNL and edge results remain separate because
they use different reference distributions.  These quantities describe model
reliance, not causal ecological effects, and are not reported as exact SHAP
values.

The ECC design follows Simonovsky and Komodakis (2017), *Dynamic
Edge-Conditioned Filters in Convolutional Neural Networks on Graphs*. All
directed layers are implemented in PyTorch with `edge_index` and `index_add_`;
PyTorch Geometric is not required.

The implementation diagrams are in
[`docs/plots/seal_directed_flowcharts`](../../../docs/plots/seal_directed_flowcharts/):
the two `seal_directed_attrs_*` files document the inherited node-attributed
baseline and the two `seal_directed_edge_weight_proposal_*` files document the
edge-covariate architecture implemented here.

## Build and validate 290 enriched MAT files

```bash
PYTHON=/Users/acw792/miniconda3/envs/Foodweb/bin/python

"${PYTHON}" src/python/seal_directed_weighted/build_weighted_mats.py \
  --overwrite

"${PYTHON}" src/python/seal_directed_weighted/build_weighted_mats.py \
  --validate-only
```

Outputs are written to
`src/python/seal_directed_weighted/data/foodwebs_mat_seal_weighted/` with the
raw-source hash and `full_v1` schema hash embedded in every MAT.

## Frozen paired cohort

The default manifest creates 2,900 new splits (290 webs x 10 experiments) with
unique seeds 22345-25244. Each split file is referenced byte-for-byte by all
four variants, producing 11,600 expected runs. Negatives remain directed,
role-filtered, 1:1, without replacement, with the documented fallback to all
directed non-links when the role-compatible pool is insufficient.

```bash
"${PYTHON}" src/python/seal_directed_weighted/generate_manifest.py \
  --foodweb-csv src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv \
  --mat-folder src/python/seal_directed_weighted/data/foodwebs_mat_seal_weighted \
  --output-dir /path/to/new-run/manifest
```

No Tukey filter or outlier deletion is applied. The primary metric is PR-AUC;
the validator also records ROC-AUC, F1, MCC, Precision, Recall and TSS.

## Tests

```bash
env PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -m unittest discover \
  -s src/python/seal_directed_weighted/tests -p 'test_*.py'
```

## Apocrita sequence

Build `libgnn.so` with `/usr/bin/g++` using `apocrita/create_gpu_env.sh`.
Create and verify the deployment bundle with `apocrita/build_bundle.py`, then:

1. `submit_gpu.sh smoke` runs 3 webs x 4 variants in `gpushort`;
2. `submit_gpu.sh calibration` runs experiment 1 for all four variants;
3. inspect CUDA, GPU memory, time and the monitor CSVs;
4. `submit_gpu.sh campaign` submits 40 tasks (`4 x 10`) with `%2` concurrency;
5. validate all 11,600 bundles with `validate_run.py --require-cuda`.

Submissions are held by default. Inspect the manifest, resources and array
mapping before using `scontrol release`.

## Attribution v2 on Apocrita

The attribution campaign has its own immutable result root and reuses the
historical split bundles without editing the 11,600 original runs.  Set these
variables in a login shell after deploying the source bundle:

```bash
export REPO_ROOT=/data/home/$USER/PhD/Stage2/Code/seal-directed-node-ecc-attribution-v2
export RUN_ROOT=/data/home/$USER/PhD/Stage2/Results/SEAL_directed_node_ecc_attribution_v2/production/run_YYYYMMDD
export HISTORICAL_RUN_ROOT=/data/home/$USER/PhD/Stage2/Results/SEAL_directed_weighted/production/seal_weighted_10x290x4_fb60fd147be3_20261024
export SEAL_ENV_PREFIX=/gpfs/scratch/$USER/SEAL_directed/envs/seal-directed-py312-cu126
export SCRATCH_ROOT=/gpfs/scratch/$USER/SEAL_directed_node_ecc_attribution_v2

LAUNCHER="$REPO_ROOT/src/python/seal_directed_weighted/apocrita/submit_node_ecc_attribution.sh"
```

The launcher refuses to run when `RUN_ROOT` resolves to the historical result
root.

Run the stages in order. Every submitted job starts in `HOLD` unless
`SUBMIT_HELD=0` is set explicitly:

```bash
"$LAUNCHER" init
"$LAUNCHER" train-smoke
"$LAUNCHER" attr-smoke

"$LAUNCHER" train-experiment1
"$LAUNCHER" calibration 32
"$LAUNCHER" calibration 64
"$LAUNCHER" calibration 128
"$LAUNCHER" calibration 256
"$LAUNCHER" freeze

"$LAUNCHER" train-campaign
"$LAUNCHER" attr-experiment1
"$LAUNCHER" attr-campaign
"$LAUNCHER" validate
"$LAUNCHER" summarize
```

`freeze` is fail-closed: it creates the production
`ATTRIBUTION_CONFIG.json` only when one common setting passes every node,
DRNL and ECC-edge convergence gate. Faithfulness replacements are evaluated
during calibration; production bundles retain the selected EG/IG setting but
do not repeat that extra control. Each stage still contains 5,800 scientific
runs (290 food webs x 10 experiments x 2 models), but the scheduler mapping
packs up to six food webs from the same model and experiment into each task.
With the default `SEAL_FOODWEB_BLOCK_SIZE=6`, the largest array index is 980,
and concurrency is controlled by `SEAL_MAX_CONCURRENT`. The block size cannot
be smaller than six under the cluster's 1,000-index array limit; increase
`SEAL_GPU_TIME` if the slowest six-web attribution blocks need more than the
default 24 hours.

The final validator requires all 2,900 bundles for each model, exact candidate
coverage, checkpoint and forward parity, CUDA provenance, finite
attributions, and an explicit historical-score comparison. Summaries are
written below `summary/attribution/`. They include separate feature- and
family-level tables for the 44 node attributes (six families), separate
feature- and family-level tables for the 17 ECC edge covariates (five
families), and an independent DRNL table. Channels are never summed together,
and every cohort summary gives equal final weight to each food web. Run this
post-processing on a compute node rather than a login node: `validate` and
`summarize` submit separate held high-memory jobs. Candidate matrices are
retained for one model variant at a time and can still require several GB for
the complete campaign.

## Descriptive analysis

`summarize_descriptive.py` averages the ten raw runs per food web, computes
paired contrasts, and produces ecosystem-stratified bootstrap 95% confidence
intervals with 10,000 replicates. It deliberately reports no p-values. Results
must also be stratified by negative-pool fallback and attribute missingness
before deciding whether ECC justifies its added complexity.
`summarize_resources.py` separately reports run time, subgraph-extraction time,
training time, GPU utilization and peak recorded GPU memory from the preserved
`nvidia-smi` traces.
