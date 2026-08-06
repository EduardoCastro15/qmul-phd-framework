#!/bin/bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./launch_wlnm_original_train90.sh smoke
  ./launch_wlnm_original_train90.sh smoke-seven
  ./launch_wlnm_original_train90.sh full

Optional for the full run:
  WLNM_ARRAY_CONCURRENCY=8 ./launch_wlnm_original_train90.sh full

The array is submitted in HOLD state. Inspect the manifest and submission
file, then release the parent job explicitly with scontrol release.
USAGE
}

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "smoke-seven" && "$mode" != "full" ]]; then
    usage >&2
    exit 64
fi

template="sbatch_wlnm_original_array_50.sbatch"
foodweb_csv="data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
slurm_log_dir="../../slurm_logs"
source_commit="${WLNM_SOURCE_COMMIT:-$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)}"
condition_id="original_legacy_uppertriangular_train90_thresh0p50"

if [[ ! -f "$template" ]]; then
    echo "ERROR: Missing template: $template" >&2
    exit 66
fi

if [[ ! -f "$foodweb_csv" ]]; then
    echo "ERROR: Missing food-web index: $foodweb_csv" >&2
    exit 66
fi

foodweb_count=$(( $(wc -l < "$foodweb_csv") - 1 ))
if [[ "$foodweb_count" -ne 290 ]]; then
    echo "ERROR: Expected 290 food webs, found ${foodweb_count}." >&2
    exit 65
fi

mkdir -p "$slurm_log_dir"

if [[ "$mode" == "smoke" ]]; then
    output_root="data/result_smoke_wlnm_original_legacy_uppertriangular_train90_thresh0p50"
    job_name="SMK_ORIG_LEGACY"
    num_experiments=1
    parallel_workers=1
    resource_args=(
        --array=1-1%1
        --ntasks=2
        --partition=compute
        --mem-per-cpu=4G
        --time=02:00:00
    )
    expected_prediction_csvs=1
    expected_terminal_logs=1
    expected_rows_per_csv=1
    expected_pool_limited_foodwebs=0
    array_concurrency=1
    selected_foodweb_indices="1"
elif [[ "$mode" == "smoke-seven" ]]; then
    output_root="data/result_smoke7_wlnm_original_legacy_uppertriangular_train90_thresh0p50"
    job_name="SMK7_ORIG_LEGACY"
    num_experiments=1
    parallel_workers=1
    resource_args=(
        --array=98,117,154,156,180,190,206%7
        --ntasks=2
        --partition=compute
        --mem-per-cpu=4G
        --time=02:00:00
    )
    expected_prediction_csvs=7
    expected_terminal_logs=7
    expected_rows_per_csv=1
    expected_pool_limited_foodwebs=0
    array_concurrency=7
    selected_foodweb_indices="98,117,154,156,180,190,206"
else
    output_root="data/result_wlnm_original_50x290_train90_thresh0p50_legacy_uppertriangular_checkconnfalse_adaptivefalse_Apocrita"
    job_name="WLNM_ORIG_LEG_T90"
    num_experiments=50
    parallel_workers=50
    array_concurrency="${WLNM_ARRAY_CONCURRENCY:-4}"
    if [[ ! "$array_concurrency" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: WLNM_ARRAY_CONCURRENCY must be a positive integer." >&2
        exit 64
    fi
    resource_args=(--array="1-290%${array_concurrency}")
    expected_prediction_csvs=290
    expected_terminal_logs=290
    expected_rows_per_csv=50
    expected_pool_limited_foodwebs=0
    selected_foodweb_indices="1-290"
fi

if [[ -e "$output_root" ]]; then
    echo "ERROR: Output root already exists; no job was submitted:" >&2
    echo "  $output_root" >&2
    exit 73
fi

mkdir "$output_root"

{
    echo "RunMode=${mode}"
    echo "Condition=${condition_id}"
    echo "Version=WLNM_original"
    echo "GraphMode=legacy_upper_triangle_of_directed_input"
    echo "OriginalCompatibilityMode=true"
    echo "SelfLoops=removed_during_upper_triangle_projection"
    echo "NegativeSampling=uniform_upper_triangular_nonlinks_without_replacement"
    echo "TargetNegativePositiveRatio=2"
    echo "NegativePoolPolicy=cap_at_full_pool"
    echo "SubgraphK=10"
    echo "TrainRatio=0.90"
    echo "ClassificationThreshold=0.50"
    echo "ThresholdSweep=false"
    echo "CheckConnectivity=false"
    echo "AdaptiveConnectivity=false"
    echo "BaseSeed=12345"
    echo "ResampleSplitsEachExperiment=true"
    echo "ComputeEcologicalMetrics=false"
    echo "FoodWebs=${foodweb_count}"
    echo "SelectedFoodWebIndices=${selected_foodweb_indices}"
    echo "NumExperiments=${num_experiments}"
    echo "ArrayConcurrency=${array_concurrency}"
    echo "ExpectedPredictionCSVs=${expected_prediction_csvs}"
    echo "ExpectedTerminalLogs=${expected_terminal_logs}"
    echo "ExpectedDataRowsPerCSV=${expected_rows_per_csv}"
    echo "ExpectedNegativePoolLimitedFoodWebs=${expected_pool_limited_foodwebs}"
    echo "SourceCommit=${source_commit}"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "Template=${template}"
} > "${output_root}/RUN_MANIFEST.txt"

run_id="$(date +%Y%m%d_%H%M%S)"
submission_file="${slurm_log_dir}/wlnm_original_train90_${mode}_${run_id}.tsv"
printf 'JobID\tCondition\tOutputRoot\tScript\n' > "$submission_file"

export_spec="ALL,WLNM_CONDITION_ID=${condition_id}"
export_spec+=",WLNM_OUTPUT_ROOT=${output_root}"
export_spec+=",WLNM_NUM_EXPERIMENTS=${num_experiments}"
export_spec+=",WLNM_PARALLEL_WORKERS=${parallel_workers}"
export_spec+=",WLNM_BASE_SEED=12345"
export_spec+=",WLNM_RATIO_TRAIN=0.90"
export_spec+=",WLNM_FIXED_THRESHOLD=0.50"

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

echo "Submitted ${mode} array in HOLD state."
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
