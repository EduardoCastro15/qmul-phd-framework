#!/bin/bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./launch_wlnm_dir_neg_yujie.sh smoke
  ./launch_wlnm_dir_neg_yujie.sh calibration
  ./launch_wlnm_dir_neg_yujie.sh primary60
  ./launch_wlnm_dir_neg_yujie.sh sweep
  ./launch_wlnm_dir_neg_yujie.sh final60

Every submission starts in HOLD state. Inspect RUN_MANIFEST.txt and the job
configuration before releasing it with scontrol.
USAGE
}

mode="${1:-}"
case "$mode" in
    smoke|calibration|primary60|sweep|final60) ;;
    *) usage >&2; exit 64 ;;
esac

matlab_dir="$(pwd -P)"
project_root="$(cd ../.. && pwd -P)"
derived_root="data/yujie_wlnm_preprocessed_v1"
template="sbatch_wlnm_dir_neg_yujie_observed_zero.sbatch"
slurm_log_dir="${project_root}/slurm_logs"
run_stamp="$(date +%Y%m%d_%H%M%S)"
source_commit="${WLNM_SOURCE_COMMIT:-$(git rev-parse --short=10 HEAD 2>/dev/null || echo uncommitted-snapshot)}"
result_base="${WLNM_RESULT_BASE:-data}"

case "$mode" in
    smoke)
        regime="original_observed"
        input_csv="${derived_root}/manifests/original_observed_smoke.csv"
        expected_networks=3
        num_experiments=1
        parallel_workers=1
        sweep=false
        ratio_count=1
        resource_args=(--array=1-3%2 --ntasks=2 --partition=compute --mem-per-cpu=4G --time=02:00:00)
        ;;
    calibration)
        regime="original_observed"
        input_csv="${derived_root}/manifests/original_observed_smoke.csv"
        expected_networks=3
        num_experiments=10
        parallel_workers=10
        sweep=false
        ratio_count=1
        resource_args=(--array=1-3%2 --ntasks=11 --partition=compute --mem-per-cpu=4G --time=08:00:00)
        ;;
    primary60)
        regime="original_observed"
        input_csv="${derived_root}/manifests/original_observed_train60_eligible.csv"
        expected_networks=178
        num_experiments=100
        parallel_workers=50
        sweep=false
        ratio_count=1
        resource_args=(--array=1-178%2)
        ;;
    sweep)
        regime="original_observed"
        input_csv="${derived_root}/manifests/original_observed_sweep_eligible.csv"
        expected_networks=144
        num_experiments=100
        parallel_workers=50
        sweep=true
        ratio_count=9
        resource_args=(--array=1-144%2)
        ;;
    final60)
        regime="final_filled_sensitivity"
        input_csv="${derived_root}/manifests/final_filled_sensitivity_train60_eligible.csv"
        expected_networks=545
        num_experiments=100
        parallel_workers=50
        sweep=false
        ratio_count=1
        resource_args=(--array=1-545%2)
        ;;
esac

mat_folder="${derived_root}/${regime}"
for required_file in "$template" "$input_csv" "${derived_root}/validation/preprocessing_summary.json"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: Missing required file: ${required_file}" >&2
        exit 66
    fi
done

actual_networks=$(python3 -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows or "Foodweb" not in rows[0]:
    raise SystemExit("Input manifest must contain Foodweb")
if len({row["Foodweb"] for row in rows}) != len(rows):
    raise SystemExit("Input manifest contains duplicate Foodweb values")
print(len(rows))
' "$input_csv")
if [[ "$actual_networks" -ne "$expected_networks" ]]; then
    echo "ERROR: Expected ${expected_networks} input networks, found ${actual_networks}." >&2
    exit 65
fi

manifest_hash=$(python3 -c '
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
' "$input_csv")

output_root="${result_base}/result_wlnm_dir_neg_yujie_${mode}_observedzero_${run_stamp}"
mkdir -p "$slurm_log_dir" "$result_base"
if [[ -e "$output_root" ]]; then
    echo "ERROR: Output root already exists: ${output_root}" >&2
    exit 73
fi
mkdir "$output_root"
mkdir "${output_root}/completion_markers"
cp "$input_csv" "${output_root}/INPUT_NETWORKS.csv"

expected_rows_per_csv=$((num_experiments * ratio_count))
{
    echo "RunMode=${mode}"
    echo "Version=WLNM_dir_neg"
    echo "DataRegime=${regime}"
    echo "InputManifest=${input_csv}"
    echo "InputManifestSHA256=${manifest_hash}"
    echo "ExpectedPredictionCSVs=${expected_networks}"
    echo "ExpectedTerminalLogs=${expected_networks}"
    echo "ExpectedCompletionMarkers=${expected_networks}"
    echo "ExpectedDataRowsPerCSV=${expected_rows_per_csv}"
    echo "NumExperiments=${num_experiments}"
    echo "ParallelWorkers=${parallel_workers}"
    echo "SweepTrainRatios=${sweep}"
    if [[ "$sweep" == "true" ]]; then
        echo "TrainRatioRange=10,20,30,40,50,60,70,80,90"
    else
        echo "TrainRatioRange=60"
    fi
    echo "SubgraphK=10"
    echo "Eligibility=observed_zero"
    echo "NegativePositiveRatio=2"
    echo "NegativeSampling=uniform_without_replacement"
    echo "NegativeTopupPolicy=error"
    echo "ClassificationThreshold=0.50"
    echo "Backbone=false"
    echo "CheckConnectivity=false"
    echo "CrossValidation=false"
    echo "MassEligibility=false"
    echo "GraphEncodingParallel=false"
    echo "ComputeEcologicalMetrics=false"
    echo "BaseSeed=12345"
    echo "ResampleSplits=true"
    echo "SourceCommit=${source_commit}"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "Template=${template}"
} > "${output_root}/RUN_MANIFEST.txt"

export_spec="ALL,WLNM_OUTPUT_ROOT=${output_root}"
export_spec+=",WLNM_NUM_EXPERIMENTS=${num_experiments}"
export_spec+=",WLNM_PARALLEL_WORKERS=${parallel_workers}"
export_spec+=",WLNM_FOODWEB_CSV=${input_csv}"
export_spec+=",WLNM_MAT_FOLDER=${mat_folder}"
export_spec+=",WLNM_SWEEP_TRAIN_RATIOS=${sweep}"
if [[ "$sweep" == "true" ]]; then
    export_spec+=",WLNM_TRAIN_RATIO_RANGE=0.10:0.10:0.90"
else
    export_spec+=",WLNM_RATIO_TRAIN=0.60"
fi

submission_output=$(sbatch \
    --parsable \
    --hold \
    --chdir="$matlab_dir" \
    --output="${slurm_log_dir}/%x.%A_%a.out" \
    --error="${slurm_log_dir}/%x.%A_%a.err" \
    "${resource_args[@]}" \
    --export="$export_spec" \
    "$template")
job_id="${submission_output%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Could not parse job ID from: ${submission_output}" >&2
    exit 70
fi

echo "JobID=${job_id}" >> "${output_root}/RUN_MANIFEST.txt"
printf 'JobID\tMode\tOutputRoot\tScript\n%s\t%s\t%s\t%s\n' \
    "$job_id" "$mode" "$output_root" "$template" > \
    "${slurm_log_dir}/wlnm_yujie_${mode}_${run_stamp}.tsv"

echo "Submitted ${mode} array in HOLD state."
echo "Output root: ${output_root}"
echo "Job ID: ${job_id}"
echo "Inspect: cat ${output_root}/RUN_MANIFEST.txt"
echo "Release only after review: scontrol release ${job_id}"
