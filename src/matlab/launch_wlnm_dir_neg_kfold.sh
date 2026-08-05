#!/bin/bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./launch_wlnm_dir_neg_kfold.sh smoke
  ./launch_wlnm_dir_neg_kfold.sh full

The array is submitted in HOLD state. Inspect the manifest and submission
file, then release the parent job explicitly with scontrol release.
USAGE
}

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
    usage >&2
    exit 64
fi

template="sbatch_wlnm_dir_neg_kfold_array_20.sbatch"
foodweb_csv="data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
slurm_log_dir="../../slurm_logs"
source_commit="${WLNM_SOURCE_COMMIT:-$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)}"
condition_id="kfold_cvK3-5-10_roleormass_tau0p80_thresh0p50"

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
    output_root="data/result_smoke_wlnm_dir_neg_kfold_cvK3_roleormass_tau0p80_thresh0p50"
    job_name="SMK_WLNM_DN_KF3"
    cv_k_list=3
    cv_k_manifest=3
    num_experiments=1
    parallel_workers=1
    resource_args=(
        --array=1-1%1
        --ntasks=2
        --partition=compute
        --mem-per-cpu=7G
        --time=02:00:00
    )
    expected_prediction_csvs=1
    expected_terminal_logs=1
    expected_rows_k3=3
    expected_rows_k5=0
    expected_rows_k10=0
else
    output_root="data/result_wlnm_dir_neg_kfold_20x290_cvK3-5-10_thresh0p50_randomeligible_roleormass_tau0p80_Apocrita"
    job_name="WLNM_DN_KF_T080"
    # Slurm parses commas in --export as variable separators, so pass this
    # numeric list to MATLAB with spaces and keep commas only in the manifest.
    cv_k_list="3 5 10"
    cv_k_manifest="3,5,10"
    num_experiments=20
    parallel_workers=20
    resource_args=(--array=1-290%2)
    expected_prediction_csvs=870
    expected_terminal_logs=870
    expected_rows_k3=60
    expected_rows_k5=100
    expected_rows_k10=200
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
    echo "Version=WLNM_dir_neg_kfold"
    echo "CvKList=${cv_k_manifest}"
    echo "CvSeed=12345"
    echo "CvStratifyBackbone=false"
    echo "FoldConnectivityConstraint=not_applicable"
    echo "TauMass=0.80"
    echo "MassEligibilityEnabled=true"
    echo "Eligibility=role_or_mass"
    echo "NegativeSampling=random_eligible_pool"
    echo "PrioritySampling=false"
    echo "TargetNegativePositiveRatio=2"
    echo "SubgraphK=10"
    echo "ClassificationThreshold=0.50"
    echo "ThresholdSweep=false"
    echo "BaseSeed=12345"
    echo "ComputeEcologicalMetrics=false"
    echo "FoodWebs=${foodweb_count}"
    echo "NumExperimentsPerFold=${num_experiments}"
    echo "ExpectedPredictionCSVs=${expected_prediction_csvs}"
    echo "ExpectedTerminalLogs=${expected_terminal_logs}"
    echo "ExpectedDataRowsCvK3=${expected_rows_k3}"
    echo "ExpectedDataRowsCvK5=${expected_rows_k5}"
    echo "ExpectedDataRowsCvK10=${expected_rows_k10}"
    echo "SourceCommit=${source_commit}"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "Template=${template}"
} > "${output_root}/RUN_MANIFEST.txt"

run_id="$(date +%Y%m%d_%H%M%S)"
submission_file="${slurm_log_dir}/wlnm_dir_neg_kfold_${mode}_${run_id}.tsv"
printf 'JobID\tCondition\tOutputRoot\tScript\n' > "$submission_file"

export_spec="ALL,WLNM_CONDITION_ID=${condition_id}"
export_spec+=",WLNM_OUTPUT_ROOT=${output_root}"
export_spec+=",WLNM_CV_K_LIST=${cv_k_list}"
export_spec+=",WLNM_CV_SEED=12345"
export_spec+=",WLNM_NUM_EXPERIMENTS=${num_experiments}"
export_spec+=",WLNM_PARALLEL_WORKERS=${parallel_workers}"
export_spec+=",WLNM_BASE_SEED=12345"
export_spec+=",WLNM_NEGATIVE_MASS_ELIGIBILITY_THRESHOLD=0.80"
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
