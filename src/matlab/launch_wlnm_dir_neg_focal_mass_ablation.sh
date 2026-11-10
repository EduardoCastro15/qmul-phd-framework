#!/bin/bash

set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "pilot" && "$mode" != "full" ]]; then
    echo "Usage: $0 pilot|full" >&2
    exit 64
fi

matlab_dir="$(pwd -P)"
project_root="$(cd ../.. && pwd -P)"
template="sbatch_wlnm_dir_neg_focal_mass_array.sbatch"
full_csv="data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
pilot_csv="data/foodwebs_mat/foodweb_focal_mass_pilot.csv"
result_base="${WLNM_RESULT_BASE:-data}"
slurm_log_dir="${project_root}/slurm_logs"
source_commit="${WLNM_SOURCE_COMMIT:-$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)}"
timestamp="$(date +%Y%m%d_%H%M%S)"

if [[ "$mode" == "pilot" ]]; then
    foodweb_csv="$pilot_csv"
    foodweb_count=10
    experiments=5
    workers=5
    ratios="0.60:0.30:0.90"
    ratio_count=2
    array_spec="1-10%4"
    expected_rows=10
    parent_root="${result_base}/result_wlnm_dir_neg_focal_mass_pilot_${timestamp}"
    resource_args=(--array="$array_spec" --ntasks=6 --partition=compute --mem-per-cpu=4G --time=12:00:00)
else
    foodweb_csv="$full_csv"
    foodweb_count=290
    experiments=100
    workers=100
    ratios="0.10:0.10:0.90"
    ratio_count=9
    array_spec="1-290%4"
    expected_rows=900
    parent_root="${result_base}/result_wlnm_dir_neg_focal_mass_full_${timestamp}"
    resource_args=(--array="$array_spec")
fi

for required in "$template" "$foodweb_csv"; do
    [[ -f "$required" ]] || { echo "ERROR: Missing ${required}" >&2; exit 66; }
done
actual_count=$(( $(wc -l < "$foodweb_csv") - 1 ))
[[ "$actual_count" -eq "$foodweb_count" ]] || {
    echo "ERROR: Expected ${foodweb_count} food webs, found ${actual_count}." >&2
    exit 65
}
[[ ! -e "$parent_root" ]] || { echo "ERROR: Output root exists: ${parent_root}" >&2; exit 73; }

mkdir -p "$slurm_log_dir" "$parent_root"
{
    echo "RunMode=${mode}"
    echo "Version=WLNM_dir_neg_focal_mass"
    echo "FoodWebCSV=${foodweb_csv}"
    echo "FoodWebs=${foodweb_count}"
    echo "NumExperiments=${experiments}"
    echo "TrainRatioRange=${ratios}"
    echo "SubgraphK=10"
    echo "ExpectedModels=$((foodweb_count * experiments * ratio_count * 4))"
    echo "SourceCommit=${source_commit}"
    echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "SubmissionState=HOLD"
} > "${parent_root}/ABLATION_MANIFEST.txt"

variants=(
    "A_structure_only:structure_only:0:90"
    "B_endpoint_masses:endpoint_masses:4:94"
    "C_mass_ratio:mass_ratio:2:92"
    "D_endpoint_masses_ratio:endpoint_masses_ratio:5:95"
)

submission_file="${slurm_log_dir}/wlnm_focal_mass_${mode}_${timestamp}.tsv"
printf 'JobID\tVariant\tFeatureMode\tOutputRoot\n' > "$submission_file"

for spec in "${variants[@]}"; do
    IFS=: read -r variant feature_mode feature_dim model_dim <<< "$spec"
    variant_root="${parent_root}/${variant}"
    mkdir -p "${variant_root}/completion_markers"
    {
        echo "RunMode=${mode}"
        echo "Variant=${variant}"
        echo "Version=WLNM_dir_neg_focal_mass"
        echo "FocalMassFeatureMode=${feature_mode}"
        echo "FocalMassFeatureDim=${feature_dim}"
        echo "ModelInputDim=${model_dim}"
        echo "MassPreprocessingProtocol=log10_within_web_median_iqr_v1"
        echo "TargetMaskingProtocol=historical_weighted_v1"
        echo "EcosystemTypeModelFeature=false"
        echo "NegativeEligibilityMode=role_only"
        echo "NegativePositiveRatio=2"
        echo "NegativeSamplingStrategy=uniform_without_replacement"
        echo "NegativeTopupPolicy=uniform_remaining_nonlinks"
        echo "NegativeMassEligibilityEnabled=false"
        echo "FoodWebs=${foodweb_count}"
        echo "NumExperiments=${experiments}"
        echo "TrainRatioRange=${ratios}"
        echo "ExpectedPredictionCSVs=${foodweb_count}"
        echo "ExpectedTerminalLogs=${foodweb_count}"
        echo "ExpectedCompletionMarkers=${foodweb_count}"
        echo "ExpectedRowsPerCSV=${expected_rows}"
        echo "SourceCommit=${source_commit}"
    } > "${variant_root}/RUN_MANIFEST.txt"

    export_spec="ALL,WLNM_OUTPUT_ROOT=${variant_root}"
    export_spec+=",WLNM_FOCAL_MASS_FEATURE_MODE=${feature_mode}"
    export_spec+=",WLNM_NUM_EXPERIMENTS=${experiments}"
    export_spec+=",WLNM_PARALLEL_WORKERS=${workers}"
    export_spec+=",WLNM_TRAIN_RATIO_RANGE=${ratios}"
    export_spec+=",WLNM_FOODWEB_CSV=${foodweb_csv}"

    submission=$(sbatch --parsable --hold --chdir="$matlab_dir" \
        --output="${slurm_log_dir}/%x.%A_%a.out" \
        --error="${slurm_log_dir}/%x.%A_%a.err" \
        --job-name="FM_${variant:0:1}_${mode}" \
        --export="$export_spec" "${resource_args[@]}" "$template")
    job_id="${submission%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "ERROR: Invalid sbatch response: ${submission}" >&2; exit 70; }
    echo "JobID=${job_id}" >> "${variant_root}/RUN_MANIFEST.txt"
    printf '%s\t%s\t%s\t%s\n' "$job_id" "$variant" "$feature_mode" "$variant_root" >> "$submission_file"
done

echo "Submitted four ${mode} ablation arrays in HOLD state."
echo "Parent root: ${parent_root}"
echo "Submission ledger: ${submission_file}"
echo "Inspect manifests before release: scontrol release <job-id>"
echo "After completion: python3 validate_wlnm_dir_neg_focal_mass_outputs.py ${parent_root}"
