#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

foodweb_csv="data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
foodweb_index=216
foodweb_name='Grand Caricaie  marsh dominated by Cladietum marisci, not mown  ClControl2_tax_mass'
train_ratio='0.60'
experiment_id="${WLNM_FIGURE2_EXPERIMENT_ID:-22}"
expected_seed="${WLNM_FIGURE2_EXPECTED_SEED:-1314098573}"
source_test_fn="${WLNM_FIGURE2_SOURCE_TEST_FN:-28}"
experiment_tag="$(printf '%03d' "$experiment_id")"
output_root="${WLNM_FIGURE2_OUTPUT_ROOT:-data/result_wlnm_dir_neg_roleonly_figure2_clcontrol2_train60_exp022_bestfn}"
selection_rule="${WLNM_FIGURE2_SELECTION_RULE:-minimum_TestFN_among_historical_50_runs}"
selection_population="${WLNM_FIGURE2_SELECTION_POPULATION:-50}"

if [[ ! -f "$foodweb_csv" ]]; then
    echo "ERROR: Missing food-web metadata: ${foodweb_csv}" >&2
    exit 66
fi

resolved_foodweb="$(python3 -c '
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

index = int(sys.argv[2]) - 1
if index < 0 or index >= len(rows):
    raise SystemExit(2)

print(rows[index]["Foodweb"])
' "$foodweb_csv" "$foodweb_index")"

if [[ "$resolved_foodweb" != "$foodweb_name" ]]; then
    echo "ERROR: Food-web index ${foodweb_index} no longer resolves to the expected dataset." >&2
    echo "Expected: ${foodweb_name}" >&2
    echo "Observed: ${resolved_foodweb}" >&2
    exit 65
fi

if [[ -e "$output_root" ]]; then
    echo "ERROR: Output root already exists; nothing was overwritten:" >&2
    echo "  ${output_root}" >&2
    echo "Set WLNM_FIGURE2_OUTPUT_ROOT to a new isolated path for another run." >&2
    exit 73
fi

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
    echo "Set MATLAB_BIN=/absolute/path/to/matlab and rerun." >&2
    exit 69
fi

mkdir -p "$output_root"

manifest_path="${output_root}/RUN_MANIFEST.txt"
{
    echo "RunMode=local_figure2_single_foodweb"
    echo "Condition=roleonly_5constraints_figure2_clcontrol2_train60_exp${experiment_tag}_bestfn"
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
    echo "TrainRatio=${train_ratio}"
    echo "ClassificationThreshold=0.50"
    echo "ThresholdMode=fixed"
    echo "CheckConnectivity=false"
    echo "AdaptiveConnectivity=false"
    echo "BaseSeed=12345"
    echo "ResampleSplitsEachExperiment=true"
    echo "SelectionRule=${selection_rule}"
    echo "SelectionPopulationRuns=${selection_population}"
    echo "SourceTestFN=${source_test_fn}"
    echo "NumExperimentsConfigured=50"
    echo "ExecutedExperimentIDs=${experiment_id}"
    echo "AuxiliaryExportExperimentID=${experiment_id}"
    echo "ExpectedExportSeed=${expected_seed}"
    echo "ExportAuxiliaryCSVs=true"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "MATLAB=${matlab_bin}"
} > "$manifest_path"

export WLNM_VERSION=WLNM_dir_neg
export WLNM_OUTPUT_ROOT="$output_root"
export WLNM_FOODWEB_CSV="$foodweb_csv"
export WLNM_FOODWEB_INDEX="$foodweb_index"
export WLNM_USE_PARALLEL=false
export WLNM_NUM_EXPERIMENTS=50
export WLNM_EXPERIMENT_ID_LIST="$experiment_id"
export WLNM_PARALLEL_WORKERS=1
export WLNM_BASE_SEED=12345
export WLNM_RESAMPLE_SPLITS_EACH_EXPERIMENT=true
export WLNM_SWEEP_TRAIN_RATIOS=true
export WLNM_TRAIN_RATIO_RANGE="$train_ratio"
export WLNM_CHECK_CONNECTIVITY=false
export WLNM_ADAPTIVE_CONNECTIVITY=false
export WLNM_CV_ENABLED=false
export WLNM_EXPORT_AUXILIARY_CSVS=true
export WLNM_AUXILIARY_EXPORT_EXPERIMENT_ID="$experiment_id"
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
export WLNM_COMPUTE_ECOLOGICAL_METRICS=true
export WLNM_RUN_DELTA_TTESTS=false
export WLNM_RUN_DELTA_EQUIVALENCE=false

echo "[INFO] Running local Figure 2 artifact replay."
echo "[INFO] Food web: ${foodweb_name}"
echo "[INFO] Experiment ID: ${experiment_id}"
echo "[INFO] Expected seed: ${expected_seed}"
echo "[INFO] Output root: ${output_root}"

"$matlab_bin" -nodisplay -nosplash -batch "Main"

artifact_prefix="${foodweb_name}_K_10_random_ratio60_wlnm_dir_neg_bb000_exp${experiment_tag}_seed${expected_seed}"
artifact_dir="${output_root}/confusion_matrix_csv"
missing_artifacts=0

for suffix in \
    scores_labels \
    TP_links \
    FP_links \
    FN_links \
    train_links \
    predicted_links; do
    artifact_path="${artifact_dir}/${artifact_prefix}_${suffix}.csv"
    if [[ ! -f "$artifact_path" ]]; then
        echo "ERROR: Missing expected artifact: ${artifact_path}" >&2
        missing_artifacts=$((missing_artifacts + 1))
    fi
done

if [[ "$missing_artifacts" -ne 0 ]]; then
    echo "RunStatus=FAILED_ARTIFACT_VALIDATION" >> "$manifest_path"
    exit 74
fi

result_csv="${output_root}/prediction_scores_logs/${foodweb_name}_results_random_wlnm_dir_neg.csv"
python3 -c '
import csv
import sys

path = sys.argv[1]
experiment_id = int(sys.argv[2])
expected_seed = int(sys.argv[3])

with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

matching = [
    row for row in rows
    if int(float(row["ExperimentID"])) == experiment_id
    and int(float(row["Seed"])) == expected_seed
    and abs(float(row["TrainRatio"]) - 60.0) < 1e-9
    and abs(float(row["Threshold"]) - 0.5) < 1e-9
]

if len(matching) != 1:
    raise SystemExit(
        "Expected exactly one matching ExperimentID/Seed/TrainRatio/Threshold row; "
        f"found {len(matching)}"
    )

row = matching[0]
print("[VALIDATION] Exported run metrics:")
for field in ("ROC_AUC", "PR_AUC", "F1Score", "TestMCC", "Precision", "Recall", "TestTSS"):
    print(f"  {field}={row[field]}")
' "$result_csv" "$experiment_id" "$expected_seed"

{
    echo "CompletedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "RunStatus=COMPLETE"
    echo "ArtifactPrefix=${artifact_prefix}"
    echo "ExpectedAuxiliaryFiles=6"
} >> "$manifest_path"

echo "[SUCCESS] Figure 2 inputs are ready:"
echo "  ${artifact_dir}"
echo "[SUCCESS] Manifest:"
echo "  ${manifest_path}"
