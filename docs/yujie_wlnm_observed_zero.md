# Yujie data in `WLNM_dir_neg`

This implementation treats each `fold × SG × year` as one directed diet
matrix with orientation `A(prey,predator)=1`. The source delivery under
`data/DAPSTOM – An Integrated Database & Portal for Fish Stomach Records/`
is read-only. All model inputs are generated under:

```text
src/matlab/data/yujie_wlnm_preprocessed_v1/
├── original_observed/
├── final_filled_sensitivity/
├── manifests/
└── validation/
```

## Rebuild and validate the inputs

From the repository root:

```bash
/Users/acw792/miniconda3/envs/Foodweb/bin/python \
  src/matlab/software/preprocess_yujie_wlnm.py --overwrite
```

The preprocessor verifies all 1,768 release hashes, all 855 matrix keys, and
the 831 unique candidate pairs in each evidence table. A failure in a hash,
key, orientation, duplicate pair, self-loop, node name, or state prevents the
derived release from being produced.

For each label regime, endpoints of resolved states `0/1` form the local node
universe. The MAT contains:

- `net`: positive links only;
- `observed_negative_mask`: explicit zeroes only;
- `candidate_mask`: Yujie's supplied candidate domain within the local universe;
- `unresolved_mask`: supplied candidate pairs with state `NA`;
- `taxonomy`: exact names, with guild suffixes preserved;
- `role`: canonical WLNM role derived from the complete positive matrix;
- `yujie_network_role`: original `prey_only`/`predator_only` provenance;
- `mass`: `NaN`, because mass eligibility is disabled;
- fold, SG, year, label regime, local modeled counts, both original and final
  `1/0/NA` counts, source path, and source SHA-256.

The role rule is the same as GATEWAy: `resource` has zero in-degree and
positive out-degree; `consumer` has positive in-degree and zero out-degree;
`consumer-resource` has both; `isolate` has neither. Roles are derived before
the train/test split and stay fixed across repetitions.

## Frozen negative protocol

`negativeEligibilityMode=observed_zero` is fail-closed:

- exactly two negatives are sampled per positive;
- sampling is uniform without replacement;
- every negative must be an explicit zero and inside `candidate_mask`;
- NA, self-loops, reverse/invented pairs, and arbitrary matrix non-links are excluded;
- top-up is `error`, not a fallback;
- train/test positive and negative sets are disjoint;
- `evaluate_on_all_unseen`, mass eligibility, CV, backbone, and ecological metrics are disabled.

Historical `role_only`, `role_or_mass`, `mass_only`, and `all_nonlinks` calls
retain their existing behavior and tests.

## Derived coverage

The generated manifests retain all 855 networks and their exclusion reason.

| Regime | Positive networks | Eligible at 60% | Eligible for 10–90% sweep |
|---|---:|---:|---:|
| Original observations | 235 | 178 | 144 |
| Final/imputed sensitivity | 596 | 545 | 493 |

The final regime has one otherwise 2:1-sufficient network with only one
positive. It is excluded from 60% because `DivideNet_dir_neg` would place that
positive entirely in test and leave an empty positive training graph.

## Local and Apocrita sequence

The local smoke uses the generated small, medium, and large selections in
`manifests/original_observed_smoke.csv`. The Apocrita launcher supports:

```bash
cd src/matlab
./launch_wlnm_dir_neg_yujie.sh smoke
./launch_wlnm_dir_neg_yujie.sh calibration
./launch_wlnm_dir_neg_yujie.sh primary60
./launch_wlnm_dir_neg_yujie.sh sweep
./launch_wlnm_dir_neg_yujie.sh final60
```

Every command submits its array in `HOLD` state. Review `RUN_MANIFEST.txt`, the
input manifest hash, array bounds, and resources before `scontrol release`.
The modes correspond to 1 run × 3 networks, 10 runs × 3 networks, 100 runs ×
178 original networks at 60%, 100 runs × 144 original networks × 9 ratios,
and 100 runs × 545 final/imputed networks at 60%, respectively.

## Campaign acceptance

Export accounting after the job finishes:

```bash
sacct -j JOBID --format=JobIDRaw,State,ExitCode -P -n > sacct_JOBID.txt
```

Then validate:

```bash
/Users/acw792/miniconda3/envs/Foodweb/bin/python \
  validate_wlnm_dir_neg_yujie_outputs.py RESULT_ROOT \
  --sacct-file sacct_JOBID.txt
```

The validator checks coverage, completion markers, exit states, every
`TrainRatio × ExperimentID`, source provenance, exact 2:1 balance, zero
fallbacks/shortfalls/overlap, split hashes, endpoint visibility totals, fixed
threshold, and finite predictive metrics. `--skip-sacct` is available only for
provisional content checks and is not sufficient for campaign acceptance.

The 100 repetitions are computational repetitions within a network, not 100
independent biological replicates. Preserve raw outputs before any Tukey
retention and aggregate repetitions within each network before cross-network
inference.
