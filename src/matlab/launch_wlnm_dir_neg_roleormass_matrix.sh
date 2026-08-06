#!/bin/bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./launch_wlnm_dir_neg_roleormass_matrix.sh smoke
  ./launch_wlnm_dir_neg_roleormass_matrix.sh full

The six arrays are submitted in HOLD state. Inspect the generated submission
file and release each parent job explicitly with scontrol release.
USAGE
}

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
    usage >&2
    exit 64
fi

template="sbatch_wlnm_dir_neg_roleormass_array_50.sbatch"
foodweb_csv="data/foodwebs_mat/foodweb_metrics_ecosystem.csv"
slurm_log_dir="../../slurm_logs"
source_commit="${WLNM_SOURCE_COMMIT:-unknown}"

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

condition_ids=(
    tau0p80_checkconnfalse
    tau0p90_checkconnfalse
    tau1p00_checkconnfalse
    tau0p80_checkconntrue
    tau0p90_checkconntrue
    tau1p00_checkconntrue
)
thresholds=(0.80 0.90 1.00 0.80 0.90 1.00)
connectivity=(false false false true true true)
job_names=(
    WLNM_DN_T080_C0
    WLNM_DN_T090_C0
    WLNM_DN_T100_C0
    WLNM_DN_T080_C1
    WLNM_DN_T090_C1
    WLNM_DN_T100_C1
)

output_roots=()
for condition_id in "${condition_ids[@]}"; do
    if [[ "$mode" == "smoke" ]]; then
        output_roots+=("data/result_smoke_wlnm_dir_neg_randomeligible_roleormass_${condition_id}_adaptivefalse")
    else
        output_roots+=(
            "data/result_wlnm_dir_neg_50x290_train10-90_thresh10-90_randomeligible_roleormass_${condition_id}_adaptivefalse_Apocrita"
        )
    fi
done

# Validate every destination before creating any of them.
for output_root in "${output_roots[@]}"; do
    if [[ -e "$output_root" ]]; then
        echo "ERROR: Output root already exists; no jobs were submitted:" >&2
        echo "  $output_root" >&2
        exit 73
    fi
done

run_id="$(date +%Y%m%d_%H%M%S)"
submission_file="${slurm_log_dir}/wlnm_dir_neg_roleormass_${mode}_${run_id}.tsv"
printf 'JobID\tCondition\tTauMass\tCheckConnectivity\tOutputRoot\tScript\n' > "$submission_file"

submitted_ids=()
cancel_held_jobs_on_error() {
    status=$?
    trap - ERR INT TERM
    if [[ "$status" -ne 0 && "${#submitted_ids[@]}" -gt 0 ]]; then
        echo "Submission failed; cancelling already-submitted held jobs: ${submitted_ids[*]}" >&2
        scancel "${submitted_ids[@]}" || true
    fi
    exit "$status"
}
trap cancel_held_jobs_on_error ERR INT TERM

# Reserve all six roots before submitting anything. Their existence prevents a
# second launcher invocation from targeting the same result set.
for index in "${!condition_ids[@]}"; do
    output_root="${output_roots[$index]}"
    mkdir "$output_root"

    {
        echo "RunMode=${mode}"
        echo "Condition=${condition_ids[$index]}"
        echo "TauMass=${thresholds[$index]}"
        echo "CheckConnectivity=${connectivity[$index]}"
        echo "AdaptiveConnectivity=false"
        echo "NegativeSampling=uniform_without_replacement"
        echo "Eligibility=role_or_mass"
        echo "TargetNegativePositiveRatio=2"
        echo "NegativeTopupPolicy=uniform_remaining_nonlinks"
        echo "PrioritySampling=false"
        echo "BaseSeed=12345"
        echo "SourceCommit=${source_commit}"
        echo "CreatedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
        echo "Template=${template}"
    } > "${output_root}/RUN_MANIFEST.txt"
done

for index in "${!condition_ids[@]}"; do
    condition_id="${condition_ids[$index]}"
    output_root="${output_roots[$index]}"
    job_name="${job_names[$index]}"

    if [[ "$mode" == "smoke" ]]; then
        job_name="SMK_${job_name#WLNM_DN_}"
        num_experiments=1
        parallel_workers=1
        train_ratio_range=0.60
        threshold_sweep_range=0.50
        resource_args=(
            --array=1-1%1
            --ntasks=2
            --partition=compute
            --mem-per-cpu=4G
            --time=01:00:00
        )
    else
        num_experiments=50
        parallel_workers=50
        train_ratio_range=0.10:0.10:0.90
        threshold_sweep_range=0.10:0.10:0.90
        resource_args=(--array=1-290%4)
    fi

    export_spec="ALL,WLNM_CONDITION_ID=${condition_id}"
    export_spec+=",WLNM_NEGATIVE_ELIGIBILITY_MODE=role_or_mass"
    export_spec+=",WLNM_NEGATIVE_POSITIVE_RATIO=2"
    export_spec+=",WLNM_NEGATIVE_SAMPLING_STRATEGY=uniform_without_replacement"
    export_spec+=",WLNM_NEGATIVE_TOPUP_POLICY=uniform_remaining_nonlinks"
    export_spec+=",WLNM_NEGATIVE_MASS_ELIGIBILITY_THRESHOLD=${thresholds[$index]}"
    export_spec+=",WLNM_CHECK_CONNECTIVITY=${connectivity[$index]}"
    export_spec+=",WLNM_ADAPTIVE_CONNECTIVITY=false"
    export_spec+=",WLNM_OUTPUT_ROOT=${output_root}"
    export_spec+=",WLNM_NUM_EXPERIMENTS=${num_experiments}"
    export_spec+=",WLNM_PARALLEL_WORKERS=${parallel_workers}"
    export_spec+=",WLNM_TRAIN_RATIO_RANGE=${train_ratio_range}"
    export_spec+=",WLNM_THRESHOLD_SWEEP_RANGE=${threshold_sweep_range}"

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
    submitted_ids+=("$job_id")

    echo "JobID=${job_id}" >> "${output_root}/RUN_MANIFEST.txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$job_id" "$condition_id" "${thresholds[$index]}" "${connectivity[$index]}" \
        "$output_root" "$template" >> "$submission_file"
done

trap - ERR INT TERM

job_id_csv=$(IFS=,; echo "${submitted_ids[*]}")

echo "Submitted six ${mode} arrays in HOLD state."
echo "Submission file: ${submission_file}"
echo "Job IDs: ${submitted_ids[*]}"
echo
echo "Inspect:"
echo "  column -t -s $'\\t' ${submission_file}"
echo "  squeue -j ${job_id_csv}"
echo
echo "Release after verification:"
echo "  awk 'NR > 1 {print \$1}' ${submission_file} | while read -r id; do scontrol release \"\$id\"; done"
