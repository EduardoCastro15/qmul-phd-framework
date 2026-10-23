# SEAL-directed-weighted

This package is an isolated extension of `src/python/seal_directed`. It keeps
the existing 44 node attributes and binary resource-to-consumer prediction,
then evaluates contextual edge covariates with truly directed message passing.
No scientific file in the validated `seal_directed` package is modified.

## Scientific scope

Visible name: **SEAL-directed-weighted**

Internal version: `SEAL_directed_weighted_v1`

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

The four frozen variants are:

1. `legacy_weak_node`: existing weak-projection DGCNN with node attributes;
2. `directed_node`: separate incoming/outgoing directed messages, no edge data;
3. `directed_scalar_gate`: a learned 17 -> 32 -> 1 sigmoid gate;
4. `directed_ecc`: layer- and direction-specific edge-conditioned filters.

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

## Descriptive analysis

`summarize_descriptive.py` averages the ten raw runs per food web, computes
paired contrasts, and produces ecosystem-stratified bootstrap 95% confidence
intervals with 10,000 replicates. It deliberately reports no p-values. Results
must also be stratified by negative-pool fallback and attribute missingness
before deciding whether ECC justifies its added complexity.
`summarize_resources.py` separately reports run time, subgraph-extraction time,
training time, GPU utilization and peak recorded GPU memory from the preserved
`nvidia-smi` traces.
