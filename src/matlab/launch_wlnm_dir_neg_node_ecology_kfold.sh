#!/bin/bash

set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "calibration" && "$mode" != "full" ]]; then
    echo "Usage: $0 smoke|calibration|full" >&2
    exit 64
fi

matlab_dir="$(pwd -P)"
project_root="$(cd ../.. && pwd -P)"
template="sbatch_wlnm_dir_neg_node_ecology_kfold_array.sbatch"
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

foodweb_at_index() {
    python3 -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
print(rows[int(sys.argv[2]) - 1]["Foodweb"])
' "$foodweb_csv" "$1"
}

if [[ "$(foodweb_at_index 1)" != "SF1M2_tax_mass" ]]; then
    echo "ERROR: Smoke index 1 no longer resolves to SF1M2_tax_mass." >&2
    exit 65
fi
if [[ "$(foodweb_at_index 289)" != "Weddell Sea_tax_mass" ]]; then
    echo "ERROR: Calibration index 289 no longer resolves to Weddell Sea_tax_mass." >&2
    exit 65
fi

case "$mode" in
    smoke)
        foodweb_count=1
        experiments=1
        workers=3
        cv_k_list="3"
        cv_k_manifest="3"
        array_spec="1-1%1"
        expected_csvs=1
        expected_logs=1
        expected_markers=1
        expected_rows_k3=3
        expected_rows_k5=0
        expected_rows_k10=0
        expected_models=3
        output_root="${result_base}/result_wlnm_dir_neg_node_ecology_kfold_smoke_prauc_ap_v1_${timestamp}"
        resource_args=(--array="$array_spec" --ntasks=4 --partition=compute --mem-per-cpu=8G --time=04:00:00)
        ;;
    calibration)
        foodweb_count=1
        experiments=2
        workers=36
        cv_k_list="3 5 10"
        cv_k_manifest="3,5,10"
        array_spec="289-289%1"
        expected_csvs=3
        expected_logs=3
        expected_markers=1
        expected_rows_k3=6
        expected_rows_k5=10
        expected_rows_k10=20
        expected_models=36
        output_root="${result_base}/result_wlnm_dir_neg_node_ecology_kfold_calibration_prauc_ap_v1_${timestamp}"
        resource_args=(--array="$array_spec" --ntasks=37 --partition=highmem --mem-per-cpu=4G --time=12:00:00)
        ;;
    full)
        foodweb_count=290
        experiments=100
        workers=63
        cv_k_list="3 5 10"
        cv_k_manifest="3,5,10"
        array_spec="1-290%4"
        expected_csvs=870
        expected_logs=870
        expected_markers=290
        expected_rows_k3=300
        expected_rows_k5=500
        expected_rows_k10=1000
        expected_models=522000
        output_root="${result_base}/result_wlnm_dir_neg_node_ecology_kfold_100x290_cvK3-5-10_repeatedfolds_roleonly_thresh0p50_prauc_ap_v1_Apocrita_${timestamp}"
        resource_args=(--array="$array_spec" --ntasks=64 --partition=highmem --mem-per-cpu=4G --time=72:00:00)
        ;;
esac

[[ ! -e "$output_root" ]] || {
    echo "ERROR: Output root already exists: ${output_root}" >&2
    exit 73
}
mkdir -p "$slurm_log_dir" "$output_root/completion_markers"

{
    echo "RunMode=${mode}"
    echo "Version=WLNM_dir_neg_node_ecology_kfold"
    echo "NodeEcologyProtocolVersion=WLNM_dir_neg_node_ecology"
    echo "CvPartitionProtocol=repeated_balanced_positive_edges_v1"
    echo "CvKList=${cv_k_manifest}"
    echo "CvSeed=12345"
    echo "BaseSeed=12345"
    echo "RepeatedFoldPartitions=true"
    echo "NodeEcologyFeatureDim=70"
    echo "ModelInputDim=160"
    echo "TrainingEpochs=200"
    echo "MassPreprocessingProtocol=ln_train_subgraph_median_mean_std_v1"
    echo "TargetMaskingProtocol=binary_and_weighted_candidate_mask_v1"
    echo "NodeAlignmentProtocol=canonical_node_order_node_major_padding0_v1"
    echo "NegativeRoleSource=train_positive_adjacency_v1"
    echo "FoodWebCSV=${foodweb_csv}"
    echo "FoodWebs=${foodweb_count}"
    echo "NumExperimentsPerCvK=${experiments}"
    echo "SubgraphK=10"
    echo "ClassificationThreshold=0.50"
    echo "ThresholdMode=fixed"
    echo "ThresholdSweep=false"
    echo "PRAUCProtocol=average_precision_noninterpolated_v1"
    echo "NegativeEligibilityMode=role_only"
    echo "NegativePositiveRatio=2"
    echo "NegativeSamplingStrategy=uniform_without_replacement"
    echo "NegativeTopupPolicy=uniform_remaining_nonlinks"
    echo "NegativeMassEligibilityEnabled=false"
    echo "ComputeEcologicalMetrics=false"
    echo "ParallelWorkers=${workers}"
    echo "ExpectedModels=${expected_models}"
    echo "ExpectedPredictionCSVs=${expected_csvs}"
    echo "ExpectedTerminalLogs=${expected_logs}"
    echo "ExpectedCompletionMarkers=${expected_markers}"
    echo "ExpectedDataRowsCvK3=${expected_rows_k3}"
    echo "ExpectedDataRowsCvK5=${expected_rows_k5}"
    echo "ExpectedDataRowsCvK10=${expected_rows_k10}"
    echo "SourceCommit=${source_commit}"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "SubmissionState=HOLD"
    echo "Template=${template}"
} > "${output_root}/RUN_MANIFEST.txt"

export_spec="ALL,WLNM_RUN_MODE=${mode}"
export_spec+=",WLNM_OUTPUT_ROOT=${output_root}"
export_spec+=",WLNM_CV_K_LIST=${cv_k_list}"
export_spec+=",WLNM_CV_SEED=12345"
export_spec+=",WLNM_BASE_SEED=12345"
export_spec+=",WLNM_NUM_EXPERIMENTS=${experiments}"
export_spec+=",WLNM_PARALLEL_WORKERS=${workers}"
export_spec+=",WLNM_FOODWEB_CSV=${foodweb_csv}"

submission=$(sbatch --parsable --hold --chdir="$matlab_dir" \
    --output="${slurm_log_dir}/%x.%A_%a.out" \
    --error="${slurm_log_dir}/%x.%A_%a.err" \
    --job-name="NE_KF_${mode}" \
    --export="$export_spec" "${resource_args[@]}" "$template")
job_id="${submission%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid sbatch response: ${submission}" >&2
    exit 70
fi

echo "JobID=${job_id}" >> "${output_root}/RUN_MANIFEST.txt"
submission_file="${slurm_log_dir}/wlnm_node_ecology_kfold_${mode}_${timestamp}.tsv"
printf 'JobID\tRunMode\tOutputRoot\tScript\n' > "$submission_file"
printf '%s\t%s\t%s\t%s\n' "$job_id" "$mode" "$output_root" "$template" >> "$submission_file"

echo "Submitted ${mode} node-ecology repeated-CV array in HOLD state."
echo "Output root: ${output_root}"
echo "Submission ledger: ${submission_file}"
echo "Job ID: ${job_id}"
echo "Inspect manifest, then release with: scontrol release ${job_id}"
echo "After completion: python3 validate_wlnm_dir_neg_node_ecology_kfold_outputs.py ${output_root}"
