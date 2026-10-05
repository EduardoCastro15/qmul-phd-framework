# South-west North Sea grouped-filled food webs

This release reorganizes the canonical **grouped and spatiotemporally filled** table into one candidate food-web matrix for every spatial group and year. It contains 45 groups in five fixed spatial folds, 19 years (1997-2015), and 831 candidate predator-prey pairs per group-year.

## Authoritative files

Use `fold_<n>/<group_id>/<year>/edge_evidence.csv.gz`. Its `edge_status` is the final state: 1 means present, 0 means absent given the available evidence, and blank means unresolved uncertainty. **Never convert blank NA to zero.**

The final state may be original or imputed. Always inspect `was_spatiotemporally_imputed`, `edge_status_before_imputation`, `edge_status_original_grouped`, `imputation_stage`, and the donor-audit fields before interpreting a value. `evidence_class` distinguishes `original_present`, `original_absent`, `imputed_present`, `imputed_absent`, and `insufficient_evidence`.

`positive_edges.csv` contains final state-1 edges. It retains `edge_origin` (`original` or `imputed`) and follows energy-flow direction: `source = prey`, `target = predator`.

## Important prediction warning

This complete filled table is suitable for descriptive use. Do not use it directly for retrospective prediction without reconstructing imputation inside each training fold or excluding donors later than the training cutoff. Prediction test labels must be original observed 0/1 values only.

## Scope

There are 855 group-year candidate matrices. Some are entirely unresolved. In particular, filling does not create evidence for 1997-2001. `release_manifest.csv` records final/original/imputed counts for every matrix.
