#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

foodweb_csv="data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
foodweb_index=216
foodweb_name='Grand Caricaie  marsh dominated by Cladietum marisci, not mown  ClControl2_tax_mass'
search_root="${WLNM_FIGURE2_SEARCH_ROOT:-data/result_wlnm_dir_neg_roleonly_figure2_clcontrol2_train60_search50_local}"
best_root="${WLNM_FIGURE2_BEST_ROOT:-data/result_wlnm_dir_neg_roleonly_figure2_clcontrol2_train60_bestfn_local}"
workers="${WLNM_FIGURE2_SEARCH_WORKERS:-4}"

for target in "$search_root" "$best_root"; do
    if [[ -e "$target" ]]; then
        echo "ERROR: Output root already exists; nothing was overwritten:" >&2
        echo "  ${target}" >&2
        echo "Set WLNM_FIGURE2_SEARCH_ROOT and WLNM_FIGURE2_BEST_ROOT to new paths." >&2
        exit 73
    fi
done

matlab_bin="${MATLAB_BIN:-}"
if [[ -z "$matlab_bin" ]] && command -v matlab >/dev/null 2>&1; then
    matlab_bin="$(command -v matlab)"
fi
if [[ -z "$matlab_bin" ]]; then
    for candidate in /Applications/MATLAB_R*.app/bin/matlab; do
        if [[ -x "$candidate" ]]; then
            matlab_bin="$candidate"
        fi
    done
fi
if [[ -z "$matlab_bin" || ! -x "$matlab_bin" ]]; then
    echo "ERROR: MATLAB executable not found." >&2
    exit 69
fi

mkdir -p "$search_root"
search_manifest="${search_root}/RUN_MANIFEST.txt"
{
    echo "RunMode=local_figure2_absolute_best_search"
    echo "Condition=roleonly_5constraints_figure2_clcontrol2_train60_search50"
    echo "Version=WLNM_dir_neg"
    echo "FoodwebIndex=${foodweb_index}"
    echo "Foodweb=${foodweb_name}"
    echo "Eligibility=role_only"
    echo "RoleConstraints=5"
    echo "MassEligibilityEnabled=false"
    echo "NegativeSampling=uniform_without_replacement"
    echo "TargetNegativePositiveRatio=2"
    echo "NegativeTopupPolicy=uniform_remaining_nonlinks"
    echo "SubgraphK=10"
    echo "TrainRatio=0.60"
    echo "ClassificationThreshold=0.50"
    echo "CheckConnectivity=false"
    echo "AdaptiveConnectivity=false"
    echo "BaseSeed=12345"
    echo "NumExperiments=50"
    echo "ArrayConcurrency=${workers}"
    echo "SelectionRule=min_TestFN_then_min_TestFP_then_max_F1_then_max_MCC_then_min_ExperimentID"
    echo "ExportAuxiliaryCSVs=false"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "$search_manifest"

export WLNM_VERSION=WLNM_dir_neg
export WLNM_OUTPUT_ROOT="$search_root"
export WLNM_FOODWEB_CSV="$foodweb_csv"
export WLNM_FOODWEB_INDEX="$foodweb_index"
export WLNM_USE_PARALLEL=true
export WLNM_NUM_EXPERIMENTS=50
unset WLNM_EXPERIMENT_ID_LIST
export WLNM_PARALLEL_WORKERS="$workers"
export WLNM_BASE_SEED=12345
export WLNM_RESAMPLE_SPLITS_EACH_EXPERIMENT=true
export WLNM_SWEEP_TRAIN_RATIOS=true
export WLNM_TRAIN_RATIO_RANGE=0.60
export WLNM_CHECK_CONNECTIVITY=false
export WLNM_ADAPTIVE_CONNECTIVITY=false
export WLNM_CV_ENABLED=false
export WLNM_EXPORT_AUXILIARY_CSVS=false
export WLNM_CV_SAVE_CONFUSION=false
export WLNM_THRESHOLD_MODE=fixed
export WLNM_FIXED_THRESHOLD=0.50
export WLNM_THRESHOLD_SWEEP_ENABLED=false
export WLNM_NEGATIVE_ELIGIBILITY_MODE=role_only
export WLNM_NEGATIVE_POSITIVE_RATIO=2
export WLNM_NEGATIVE_SAMPLING_STRATEGY=uniform_without_replacement
export WLNM_NEGATIVE_TOPUP_POLICY=uniform_remaining_nonlinks
export WLNM_NEGATIVE_MASS_ELIGIBILITY_ENABLED=false
export WLNM_NEGATIVE_MASS_ELIGIBILITY_THRESHOLD=1.0
export WLNM_COMPUTE_ECOLOGICAL_METRICS=false
export WLNM_RUN_DELTA_TTESTS=false
export WLNM_RUN_DELTA_EQUIVALENCE=false

echo "[STAGE 1/2] Evaluating all 50 runs with ${workers} local workers."
"$matlab_bin" -nodisplay -nosplash -batch "Main"

result_csv="${search_root}/prediction_scores_logs/${foodweb_name}_results_random_wlnm_dir_neg.csv"
selection_tsv="${search_root}/BEST_RUN_SELECTION.tsv"

selection="$({ python3 -c '
import csv
import sys

source, destination = sys.argv[1:]
with open(source, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

eligible = [
    row for row in rows
    if abs(float(row["TrainRatio"]) - 60.0) < 1e-9
    and abs(float(row["Threshold"]) - 0.5) < 1e-9
    and row.get("ThresholdMode", "fixed").strip().lower() == "fixed"
]
if len(eligible) != 50:
    raise SystemExit(f"Expected 50 eligible result rows; found {len(eligible)}")

def number(row, field):
    return float(row[field])

best = min(
    eligible,
    key=lambda row: (
        number(row, "TestFN"),
        number(row, "TestFP"),
        -number(row, "F1Score"),
        -number(row, "TestMCC"),
        int(float(row["ExperimentID"])),
    ),
)

fields = [
    "ExperimentID", "Seed", "TestTP", "TestFP", "TestFN", "TestTN",
    "Precision", "Recall", "F1Score", "TestMCC", "TestTSS", "ROC_AUC", "PR_AUC",
]
with open(destination, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow({field: best[field] for field in fields})

print("\t".join([best["ExperimentID"], best["Seed"], best["TestFN"]]))
' "$result_csv" "$selection_tsv"; })"

IFS=$'\t' read -r best_experiment_id best_seed best_fn <<< "$selection"
{
    echo "CompletedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "RunStatus=COMPLETE"
    echo "SelectedExperimentID=${best_experiment_id}"
    echo "SelectedSeed=${best_seed}"
    echo "SelectedTestFN=${best_fn}"
} >> "$search_manifest"

echo "[SELECTION] ExperimentID=${best_experiment_id}, Seed=${best_seed}, FN=${best_fn}"
echo "[STAGE 2/2] Replaying the selected run and exporting Figure 2 matrices."

export MATLAB_BIN="$matlab_bin"
export WLNM_FIGURE2_EXPERIMENT_ID="$best_experiment_id"
export WLNM_FIGURE2_EXPECTED_SEED="$best_seed"
export WLNM_FIGURE2_SOURCE_TEST_FN="$best_fn"
export WLNM_FIGURE2_SELECTION_RULE='min_TestFN_then_min_TestFP_then_max_F1_then_max_MCC_then_min_ExperimentID'
export WLNM_FIGURE2_SELECTION_POPULATION=50
export WLNM_FIGURE2_OUTPUT_ROOT="$best_root"

./launch_wlnm_dir_neg_roleonly_figure2_local.sh

python3 -c '
import csv
import sys

selection_path, replay_path = sys.argv[1:]
with open(selection_path, newline="", encoding="utf-8") as handle:
    selected = next(csv.DictReader(handle, delimiter="\t"))
with open(replay_path, newline="", encoding="utf-8") as handle:
    replayed = next(csv.DictReader(handle))

fields = ["ExperimentID", "Seed", "TestTP", "TestFP", "TestFN", "TestTN"]
mismatches = [field for field in fields if float(selected[field]) != float(replayed[field])]
if mismatches:
    raise SystemExit(
        "Selected run was not reproduced exactly for: " + ", ".join(mismatches)
    )
print("[VALIDATION] The exported matrices exactly reproduce the selected absolute-best run.")
' "$selection_tsv" \
  "${best_root}/prediction_scores_logs/${foodweb_name}_results_random_wlnm_dir_neg.csv"

echo "[SUCCESS] Absolute-best Figure 2 result:"
echo "  ${best_root}"
