#!/bin/bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./launch_wlnm_dir_neg_roleonly.sh smoke
  ./launch_wlnm_dir_neg_roleonly.sh full

The array is submitted in HOLD state. The smoke run uses one food web with
enough role candidates and one requiring random top-up. Inspect the manifest
before releasing the parent job explicitly with scontrol release.
USAGE
}

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
    usage >&2
    exit 64
fi

template="sbatch_wlnm_dir_neg_roleonly_array_50.sbatch"
foodweb_csv="data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
pool_audit_csv="data/foodwebs_mat/foodweb_negative_constraint_pool_status.csv"
slurm_log_dir="../../slurm_logs"
source_commit="${WLNM_SOURCE_COMMIT:-$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)}"
condition_id="roleonly_topup_train10-90_thresh0p50_checkconnfalse_adaptivefalse"

for required_file in "$template" "$foodweb_csv" "$pool_audit_csv"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: Missing required file: ${required_file}" >&2
        exit 66
    fi
done

foodweb_count=$(( $(wc -l < "$foodweb_csv") - 1 ))
if [[ "$foodweb_count" -ne 290 ]]; then
    echo "ERROR: Expected 290 food webs, found ${foodweb_count}." >&2
    exit 65
fi

read -r audit_count audit_sufficient audit_topup <<< "$(python3 -c '
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
sufficient = sum(row["has_sufficient_role_constrained_pool"].strip().lower() == "true" for row in rows)
topup = sum(row["requires_random_topup"].strip().lower() == "true" for row in rows)
print(len(rows), sufficient, topup)
' "$pool_audit_csv")"
if [[ "$audit_count" -ne 290 || "$audit_sufficient" -ne 207 || "$audit_topup" -ne 83 ]]; then
    echo "ERROR: Role-pool audit no longer matches the validated 207 sufficient / 83 top-up split." >&2
    echo "  rows=${audit_count} sufficient=${audit_sufficient} topup=${audit_topup}" >&2
    exit 65
fi

mkdir -p "$slurm_log_dir"

if [[ "$mode" == "smoke" ]]; then
    output_root="data/result_smoke_wlnm_dir_neg_roleonly_topup_train60_thresh0p50_checkconnfalse_adaptivefalse"
    job_name="SMK_DN_ROLE"
    num_experiments=1
    parallel_workers=1
    train_ratio_range=0.60
    array_spec="1,3%2"
    resource_args=(
        --array="$array_spec"
        --ntasks=2
        --partition=compute
        --mem-per-cpu=4G
        --time=02:00:00
    )
    expected_prediction_csvs=2
    expected_terminal_logs=2
    expected_completion_markers=2
    expected_rows_per_csv=1
    expected_topup_foodwebs=1
    expected_no_topup_foodwebs=1
else
    output_root="data/result_wlnm_dir_neg_roleonly_topup_50x290_train10-90_thresh0p50_checkconnfalse_adaptivefalse_Apocrita"
    job_name="WLNM_DN_ROLE"
    num_experiments=50
    parallel_workers=50
    train_ratio_range=0.10:0.10:0.90
    array_spec="1-290%4"
    resource_args=(--array="$array_spec")
    expected_prediction_csvs=290
    expected_terminal_logs=290
    expected_completion_markers=290
    expected_rows_per_csv=450
    expected_topup_foodwebs=83
    expected_no_topup_foodwebs=207
fi

if [[ -e "$output_root" ]]; then
    echo "ERROR: Output root already exists; no job was submitted:" >&2
    echo "  $output_root" >&2
    exit 73
fi

mkdir "$output_root"
mkdir "${output_root}/completion_markers"

{
    echo "RunMode=${mode}"
    echo "Condition=${condition_id}"
    echo "Version=WLNM_dir_neg"
    echo "Eligibility=role_only"
    echo "RoleConstraints=5"
    echo "MassEligibilityEnabled=false"
    echo "NegativeSampling=uniform_without_replacement"
    echo "TargetNegativePositiveRatio=2"
    echo "NegativeTopupPolicy=uniform_remaining_nonlinks"
    echo "SubgraphK=10"
    echo "TrainRatioRange=${train_ratio_range}"
    echo "ClassificationThreshold=0.50"
    echo "ThresholdMode=fixed"
    echo "ThresholdSweep=false"
    echo "CheckConnectivity=false"
    echo "AdaptiveConnectivity=false"
    echo "BaseSeed=12345"
    echo "ResampleSplitsEachExperiment=true"
    echo "ComputeEcologicalMetrics=true"
    echo "FoodWebs=${foodweb_count}"
    echo "ArraySpec=${array_spec}"
    echo "NumExperimentsPerTrainRatio=${num_experiments}"
    echo "ExpectedPredictionCSVs=${expected_prediction_csvs}"
    echo "ExpectedTerminalLogs=${expected_terminal_logs}"
    echo "ExpectedCompletionMarkers=${expected_completion_markers}"
    echo "ExpectedDataRowsPerCSV=${expected_rows_per_csv}"
    echo "ExpectedTopupFoodWebs=${expected_topup_foodwebs}"
    echo "ExpectedNoTopupFoodWebs=${expected_no_topup_foodwebs}"
    echo "SourceCommit=${source_commit}"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "Template=${template}"
} > "${output_root}/RUN_MANIFEST.txt"

run_id="$(date +%Y%m%d_%H%M%S)"
submission_file="${slurm_log_dir}/wlnm_dir_neg_roleonly_${mode}_${run_id}.tsv"
printf 'JobID\tCondition\tOutputRoot\tScript\n' > "$submission_file"

export_spec="ALL,WLNM_CONDITION_ID=${condition_id}"
export_spec+=",WLNM_OUTPUT_ROOT=${output_root}"
export_spec+=",WLNM_NUM_EXPERIMENTS=${num_experiments}"
export_spec+=",WLNM_PARALLEL_WORKERS=${parallel_workers}"
export_spec+=",WLNM_TRAIN_RATIO_RANGE=${train_ratio_range}"

submission_output=$(sbatch \
    --parsable \
    --hold \
    --job-name="$job_name" \
    --export="$export_spec" \
    "${resource_args[@]}" \
    "$template")
job_id="${submission_output%%;*}"

if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Could not parse job ID from: ${submission_output}" >&2
    exit 70
fi

echo "JobID=${job_id}" >> "${output_root}/RUN_MANIFEST.txt"
printf '%s\t%s\t%s\t%s\n' \
    "$job_id" "$condition_id" "$output_root" "$template" >> "$submission_file"

echo "Submitted ${mode} role-only array in HOLD state."
echo "Submission file: ${submission_file}"
echo "Output root: ${output_root}"
echo "Job ID: ${job_id}"
echo
echo "Inspect:"
echo "  cat ${output_root}/RUN_MANIFEST.txt"
echo "  squeue -j ${job_id}"
echo
echo "Release after verification:"
echo "  scontrol release ${job_id}"
