#!/bin/bash

set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
    echo "Usage: $0 smoke|full" >&2
    exit 64
fi

matlab_dir="$(pwd -P)"
project_root="$(cd ../.. && pwd -P)"
template="sbatch_wlnm_dir_neg_node_ecology_array.sbatch"
foodweb_csv="data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
result_base="${WLNM_RESULT_BASE:-data}"
slurm_log_dir="${project_root}/slurm_logs"
source_commit="${WLNM_SOURCE_COMMIT:-$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)}"
timestamp="$(date +%Y%m%d_%H%M%S)"

for required in "$template" "$foodweb_csv"; do
    [[ -f "$required" ]] || { echo "ERROR: Missing ${required}" >&2; exit 66; }
done

full_foodweb_count=$(( $(wc -l < "$foodweb_csv") - 1 ))
if [[ "$full_foodweb_count" -ne 290 ]]; then
    echo "ERROR: Expected 290 food webs, found ${full_foodweb_count}." >&2
    exit 65
fi

first_foodweb=$(python3 -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    print(next(csv.DictReader(handle))["Foodweb"])
' "$foodweb_csv")
if [[ "$first_foodweb" != "SF1M2_tax_mass" ]]; then
    echo "ERROR: Smoke task 1 must resolve SF1M2_tax_mass, got ${first_foodweb}." >&2
    exit 65
fi

if [[ "$mode" == "smoke" ]]; then
    foodweb_count=1
    experiments=1
    workers=1
    ratios="0.60"
    ratio_count=1
    array_spec="1-1%1"
    expected_rows=1
    output_root="${result_base}/result_wlnm_dir_neg_node_ecology_smoke_${timestamp}"
    resource_args=(--array="$array_spec" --ntasks=2 --partition=compute --mem-per-cpu=4G --time=02:00:00)
else
    foodweb_count=290
    experiments=100
    # Apocrita highmem provides at most 64 CPUs on one node. Reserve one
    # process for the MATLAB client and use the remaining CPUs as workers.
    workers=63
    ratios="0.10:0.10:0.90"
    ratio_count=9
    array_spec="1-290%4"
    expected_rows=900
    output_root="${result_base}/result_wlnm_dir_neg_node_ecology_full_${timestamp}"
    resource_args=(--array="$array_spec")
fi

[[ ! -e "$output_root" ]] || {
    echo "ERROR: Output root already exists: ${output_root}" >&2
    exit 73
}
mkdir -p "$slurm_log_dir" "$output_root/completion_markers"

{
    echo "RunMode=${mode}"
    echo "Version=WLNM_dir_neg_node_ecology"
    echo "NodeEcologyFeatureDim=70"
    echo "ModelInputDim=160"
    echo "TrainingEpochs=200"
    echo "MassPreprocessingProtocol=ln_train_subgraph_median_mean_std_v1"
    echo "TargetMaskingProtocol=binary_and_weighted_candidate_mask_v1"
    echo "NodeAlignmentProtocol=canonical_node_order_node_major_padding0_v1"
    echo "NegativeRoleSource=train_positive_adjacency_v1"
    echo "FoodWebCSV=${foodweb_csv}"
    echo "FoodWebs=${foodweb_count}"
    echo "NumExperiments=${experiments}"
    echo "TrainRatioRange=${ratios}"
    echo "SubgraphK=10"
    echo "ClassificationThreshold=0.50"
    echo "ThresholdMode=fixed"
    echo "ThresholdSweep=false"
    echo "NegativeEligibilityMode=role_only"
    echo "NegativePositiveRatio=2"
    echo "NegativeSamplingStrategy=uniform_without_replacement"
    echo "NegativeTopupPolicy=uniform_remaining_nonlinks"
    echo "NegativeMassEligibilityEnabled=false"
    echo "ExpectedModels=$((foodweb_count * experiments * ratio_count))"
    echo "ExpectedPredictionCSVs=${foodweb_count}"
    echo "ExpectedTerminalLogs=${foodweb_count}"
    echo "ExpectedCompletionMarkers=${foodweb_count}"
    echo "ExpectedRowsPerCSV=${expected_rows}"
    echo "SourceCommit=${source_commit}"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "SubmissionState=HOLD"
} > "${output_root}/RUN_MANIFEST.txt"

export_spec="ALL,WLNM_RUN_MODE=${mode}"
export_spec+=",WLNM_OUTPUT_ROOT=${output_root}"
export_spec+=",WLNM_NUM_EXPERIMENTS=${experiments}"
export_spec+=",WLNM_PARALLEL_WORKERS=${workers}"
export_spec+=",WLNM_TRAIN_RATIO_RANGE=${ratios}"
export_spec+=",WLNM_FOODWEB_CSV=${foodweb_csv}"

submission=$(sbatch --parsable --hold --chdir="$matlab_dir" \
    --output="${slurm_log_dir}/%x.%A_%a.out" \
    --error="${slurm_log_dir}/%x.%A_%a.err" \
    --job-name="NE_${mode}" \
    --export="$export_spec" "${resource_args[@]}" "$template")
job_id="${submission%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid sbatch response: ${submission}" >&2
    exit 70
fi

echo "JobID=${job_id}" >> "${output_root}/RUN_MANIFEST.txt"
submission_file="${slurm_log_dir}/wlnm_node_ecology_${mode}_${timestamp}.tsv"
printf 'JobID\tRunMode\tOutputRoot\tScript\n' > "$submission_file"
printf '%s\t%s\t%s\t%s\n' "$job_id" "$mode" "$output_root" "$template" >> "$submission_file"

echo "Submitted ${mode} node-ecology array in HOLD state."
echo "Output root: ${output_root}"
echo "Submission ledger: ${submission_file}"
echo "Job ID: ${job_id}"
echo "Inspect manifest, then release with: scontrol release ${job_id}"
echo "After completion: python3 validate_wlnm_dir_neg_node_ecology_outputs.py ${output_root}"
