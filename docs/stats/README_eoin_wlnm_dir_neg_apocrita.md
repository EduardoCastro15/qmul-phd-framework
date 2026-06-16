# Eoin-style WLNM_dir_neg statistics

This analysis uses the existing Apocrita sweep output:

`src/matlab/data/result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_Apocrita`

The Python preparation script filters the existing prediction logs to `TrainRatio=90` and compares each generated pseudo web with its original empirical food web. The 20 repeated WLNM runs are not averaged; they are retained as run-level pairs (`web x run_id`), as clarified by Athen.

Run from the repository root:

```sh
python3 docs/stats/prepare_eoin_wlnm_dir_neg_apocrita_train90.py
```

Outputs are written to:

`src/matlab/data/result_wlnm_dir_neg_sweep_train_ratios_10-90_pseudo_properties_Apocrita/statistical_tests/eoin`

The Eoin-named input tables are:

- `eoin_connectance_paired_input_train90.csv`: connectance-only paired t-test input with columns `web`, `ecosystem`, `real`, and `pseudo`.
- `eoin_paired_input_train90.csv`: paired t-test input for all metrics with columns `web`, `ecosystem`, `metric`, `real`, and `pseudo`.
- `eoin_mixed_model_wide_train90.csv`: mixed-model input matching Eoin's suggested structure, with columns `web`, `web_type`, `ecosystem`, `connectance`, `average_trophic_height`, `mean_generality`, and `mean_vulnerability`.
- `eoin_mixed_model_long_train90.csv`: long mixed-model input used by the R script, with columns `web`, `web_type`, `ecosystem`, `metric`, and `value`.

All four input tables include `run_id` so each pseudo web replicate can be traced to its original empirical web.

The paired t-test result tables are:

- `eoin_paired_ttest_general_results_train90.csv`: overall paired t-tests by metric.
- `eoin_paired_ttest_by_ecosystem_results_train90.csv`: ecosystem-specific paired t-tests by metric.

When R is available, run:

```sh
Rscript docs/stats/eoin_mixed_effects_wlnm_dir_neg_apocrita.R
```

The R script uses `nlme::lme(value ~ web_type * ecosystem, random = ~1 | web)` for each metric and writes ANOVA and coefficient tables into the same output directory.

The R output files are:

- `eoin_r_paired_ttest_general_results_train90.csv`
- `eoin_r_lme_anova_train90.csv`
- `eoin_r_lme_coefficients_train90.csv`
- `eoin_r_lme_posthoc_webtype_by_ecosystem_train90.csv`
- `eoin_r_session_info.txt`

If `emmeans` is installed, the R script also writes:

`eoin_r_lme_posthoc_webtype_by_ecosystem_train90.csv`

This file contains post-hoc `real - pseudo` contrasts within each ecosystem. This is the mixed-model equivalent of the follow-up step suggested by Eoin for identifying which ecosystem types exhibit empirical-pseudo differences when the `WebType:EcosystemType` interaction is significant.
