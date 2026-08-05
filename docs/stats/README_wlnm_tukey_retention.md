# WLNM Tukey run-retention outputs

`apply_wlnm_tukey_retention.py` applies the final metric-wise run-retention
protocol without modifying the original WLNM prediction CSVs.

## Frozen primary protocol

- Classification threshold: `0.5`.
- Tukey fences: `Q1 - 1.5 * IQR` and `Q3 + 1.5 * IQR`.
- Fences are independent for each result root and each
  `Foodweb x Version x TrainRatio x K x CvK x Metric` group.
- All observations inside the inclusive fences are retained.
- Runs are never ranked or selected by predictive performance.
- Minimum retention is 50% of the expected independent repetitions:
  25 of 50 standard runs and 10 of 20 repeated-CV experiments.
- K-fold observations are first averaged across a complete set of folds within
  each repeated-CV experiment. Folds are not treated as independent runs.
- Retention is metric-specific. A run can be retained for one metric and
  excluded for another.

## Generated layout

Each input result root receives:

```text
retention_protocol/
└── tukey_iqr_1p5_min50pct_threshold0p50_v1/
    ├── retention_manifest.json
    ├── retained_run_metrics.csv.gz
    ├── excluded_run_metrics.csv.gz
    ├── retained_run_ids.csv.gz
    ├── excluded_run_ids.csv.gz
    ├── retention_by_foodweb_metric.csv
    ├── retention_by_metric.csv
    ├── retained_foodwebs_by_metric.csv
    ├── validation_report.csv
    └── figure_inputs/
        ├── all_foodweb_metric_means_after_tukey.csv
        ├── predictive_train10-90_after_tukey.csv
        ├── predictive_train90_after_tukey.csv
        └── ecological_train{40,50,60,90}_after_tukey.csv
```

The compressed run-level tables contain the complete identifiers, metric
values, fences, retention counts, and exclusion reasons. The figure-input
tables contain one retained mean per food web, metric, and experimental
condition.

## Processing the local role-or-mass matrix

Run from the repository root:

```bash
python3 docs/stats/apply_wlnm_tukey_retention.py \
  --discover-role-or-mass
```

The command refuses to overwrite an existing protocol directory. Use a new
`--output-name` when intentionally creating a revised protocol.

Future corrected `WLNM_original` and `WLNM_dir_neg_kfold` roots can be
processed independently with repeated `--result-root` arguments. Legacy
`masspref` roots must not be used for final figures.
