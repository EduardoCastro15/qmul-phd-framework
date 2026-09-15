#!/bin/bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./launch_wlnm_dir_neg_trophic100.sh smoke
  ./launch_wlnm_dir_neg_trophic100.sh full

Submits the 290 food webs for the 10%-90% train-ratio sweep with validated_v2.
The full run uses 100 repeated experiments per food web and train ratio. The
smoke run remains limited to PlotB and Ythan. Submission starts in HOLD state.
USAGE
}

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
    usage >&2
    exit 64
fi

matlab_dir="$(pwd -P)"
project_root="$(cd ../.. && pwd -P)"
template="sbatch_wlnm_dir_neg_trophic100_array.sbatch"
foodweb_csv="data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
slurm_log_dir="${project_root}/slurm_logs"
run_stamp="$(date +%Y%m%d_%H%M%S)"
source_commit="${WLNM_SOURCE_COMMIT:-$(git rev-parse --short=10 HEAD 2>/dev/null || echo uncommitted-snapshot)}"
result_base="${WLNM_RESULT_BASE:-data}"

for required_file in "$template" "$foodweb_csv"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: Missing required file: ${required_file}" >&2
        exit 66
    fi
done

python3 - "$foodweb_csv" <<'PY'
import csv
import sys
expected = {
    210: "Ythan Estuary_tax_mass",
    215: "Dutch Microfauna food web PlotB_tax_mass",
}
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
for index, name in expected.items():
    if index > len(rows) or rows[index - 1]["Foodweb"] != name:
        raise SystemExit(f"Food-web index mismatch at {index}: expected {name!r}")
PY

mkdir -p "$slurm_log_dir"

if [[ "$mode" == "smoke" ]]; then
    output_root="${result_base}/result_smoke_wlnm_dir_neg_roleonly_trophicv2_1x2_train10-90_${run_stamp}"
    num_experiments=1
    parallel_workers=1
    resource_args=(--array=210,215%2 --ntasks=2 --partition=compute --mem-per-cpu=4G --time=02:00:00)
else
    output_root="${result_base}/result_wlnm_dir_neg_roleonly_trophicv2_100x290_train10-90_Apocrita_${run_stamp}"
    num_experiments=100
    parallel_workers=50
    resource_args=(--array=1-290%2)
fi

mkdir -p "$result_base"
if [[ -e "$output_root" ]]; then
    echo "ERROR: Output root already exists: ${output_root}" >&2
    exit 73
fi

mkdir "$output_root"
mkdir "${output_root}/completion_markers"
mkdir "${output_root}/ecological_snapshots"

{
    echo "RunMode=${mode}"
    echo "Version=WLNM_dir_neg"
    if [[ "$mode" == "smoke" ]]; then
        echo "FoodWebIndices=210,215"
        echo "FoodWebs=2"
    else
        echo "FoodWebIndices=1-290"
        echo "FoodWebs=290"
    fi
    echo "NumExperiments=${num_experiments}"
    echo "ParallelWorkers=${parallel_workers}"
    echo "SweepTrainRatios=true"
    echo "TrainRatioRange=10,20,30,40,50,60,70,80,90"
    echo "SubgraphK=10"
    echo "Eligibility=role_only"
    echo "MassEligibilityEnabled=false"
    echo "NegativePositiveRatio=2"
    echo "NegativeSampling=uniform_without_replacement"
    echo "NegativeTopupPolicy=uniform_remaining_nonlinks"
    echo "ClassificationThreshold=0.50"
    echo "TrophicLevelProtocol=validated_v2"
    echo "TrophicHighPrecision=auto"
    echo "ComputeEcologicalMetrics=true"
    echo "MinimumRetainedAfterTukey=25"
    echo "SourceCommit=${source_commit}"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "Template=${template}"
} > "${output_root}/RUN_MANIFEST.txt"

export_spec="ALL,WLNM_OUTPUT_ROOT=${output_root}"
export_spec+=",WLNM_NUM_EXPERIMENTS=${num_experiments}"
export_spec+=",WLNM_PARALLEL_WORKERS=${parallel_workers}"
export_spec+=",WLNM_TRAIN_RATIO_RANGE=0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90"

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
printf 'JobID\tOutputRoot\tScript\n%s\t%s\t%s\n' \
    "$job_id" "$output_root" "$template" > \
    "${slurm_log_dir}/wlnm_dir_neg_trophic100_${mode}_${run_stamp}.tsv"

echo "Submitted ${mode} trophic-v2 array in HOLD state."
echo "Output root: ${output_root}"
echo "Job ID: ${job_id}"
echo "Inspect: cat ${output_root}/RUN_MANIFEST.txt"
echo "Release: scontrol release ${job_id}"
