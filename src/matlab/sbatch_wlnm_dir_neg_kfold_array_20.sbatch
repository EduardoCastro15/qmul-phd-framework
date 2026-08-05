#!/bin/bash
#SBATCH --job-name=WLNM_DN_KF20
#SBATCH --chdir=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework-dir-neg/src/matlab
#SBATCH --output=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework-dir-neg/slurm_logs/%x.%A_%a.out
#SBATCH --error=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework-dir-neg/slurm_logs/%x.%A_%a.err
#SBATCH --array=1-290%2
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH --partition=highmem
#SBATCH --mem-per-cpu=7G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acw792@qmul.ac.uk

set -euo pipefail

: "${WLNM_CONDITION_ID:?Submit this template through launch_wlnm_dir_neg_kfold.sh}"
: "${WLNM_OUTPUT_ROOT:?Missing isolated output root}"
: "${WLNM_NEGATIVE_MASS_ELIGIBILITY_THRESHOLD:?Missing mass eligibility threshold}"

module purge
module load matlab

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WLNM_MATLAB_JOB_STORAGE="${TMPDIR:-/tmp}/wlnm_matlab_jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WLNM_MATLAB_JOB_STORAGE"

export WLNM_VERSION=WLNM_dir_neg_kfold
export WLNM_CV_ENABLED=true
export WLNM_CV_K_LIST="${WLNM_CV_K_LIST:-3,5,10}"
export WLNM_CV_SEED="${WLNM_CV_SEED:-12345}"
export WLNM_CV_STRATIFY_BACKBONE=false
export WLNM_NUM_EXPERIMENTS="${WLNM_NUM_EXPERIMENTS:-20}"
export WLNM_PARALLEL_WORKERS="${WLNM_PARALLEL_WORKERS:-20}"
export WLNM_BASE_SEED="${WLNM_BASE_SEED:-12345}"
export WLNM_SWEEP_TRAIN_RATIOS=false
export WLNM_CHECK_CONNECTIVITY=false
export WLNM_ADAPTIVE_CONNECTIVITY=false
export WLNM_THRESHOLD_MODE=fixed
export WLNM_FIXED_THRESHOLD="${WLNM_FIXED_THRESHOLD:-0.50}"
export WLNM_THRESHOLD_SWEEP_ENABLED=false
export WLNM_NEGATIVE_MASS_ELIGIBILITY_ENABLED=true
export WLNM_NEGATIVE_MASS_ELIGIBILITY_THRESHOLD
export WLNM_FOODWEB_INDEX="${SLURM_ARRAY_TASK_ID}"
export WLNM_FOODWEB_CSV=data/foodwebs_mat/foodweb_metrics_ecosystem.csv
export WLNM_OUTPUT_ROOT
export WLNM_EXPORT_AUXILIARY_CSVS=false
export WLNM_CV_SAVE_CONFUSION=false
export WLNM_RUN_DELTA_TTESTS=false
export WLNM_RUN_DELTA_EQUIVALENCE=false
export WLNM_COMPUTE_ECOLOGICAL_METRICS=false

foodweb_row=$((SLURM_ARRAY_TASK_ID + 1))
foodweb_name=$(awk -v row="$foodweb_row" '
    NR == row {
        sub(/\r$/, "", $0)
        if (substr($0, 1, 1) == "\"") {
            value = substr($0, 2)
            closing_quote = index(value, "\",")
            if (closing_quote == 0) {
                exit 2
            }
            print substr(value, 1, closing_quote - 1)
        } else {
            split($0, fields, ",")
            print fields[1]
        }
        exit
    }
' "$WLNM_FOODWEB_CSV")

if [[ -z "$foodweb_name" ]]; then
    echo "ERROR: Could not resolve food web for array task ${SLURM_ARRAY_TASK_ID}." >&2
    exit 64
fi

for cv_k in ${WLNM_CV_K_LIST//,/ }; do
    result_csv="${WLNM_OUTPUT_ROOT}/prediction_scores_logs/${foodweb_name}_results_random_wlnm_dir_neg_kfold_cvK${cv_k}.csv"
    terminal_log="${WLNM_OUTPUT_ROOT}/terminal_logs/${foodweb_name}_wlnm_dir_neg_kfold_cvK${cv_k}_terminal_log.txt"

    if [[ -e "$result_csv" || -e "$terminal_log" ]]; then
        echo "ERROR: Output collision for ${foodweb_name}, CvK=${cv_k}." >&2
        echo "Existing result: ${result_csv}" >&2
        echo "Existing terminal log: ${terminal_log}" >&2
        echo "Use an isolated retry output root instead of appending to this run." >&2
        exit 73
    fi
done

echo "Job started at: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "Running on node: $(hostname)"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "WLNM_CONDITION_ID: ${WLNM_CONDITION_ID}"
echo "WLNM_CV_K_LIST: ${WLNM_CV_K_LIST}"
echo "WLNM_CV_SEED: ${WLNM_CV_SEED}"
echo "WLNM_CV_STRATIFY_BACKBONE: ${WLNM_CV_STRATIFY_BACKBONE}"
echo "WLNM_NUM_EXPERIMENTS: ${WLNM_NUM_EXPERIMENTS}"
echo "WLNM_NEGATIVE_MASS_ELIGIBILITY_THRESHOLD: ${WLNM_NEGATIVE_MASS_ELIGIBILITY_THRESHOLD}"
echo "WLNM_FIXED_THRESHOLD: ${WLNM_FIXED_THRESHOLD}"
echo "WLNM_CHECK_CONNECTIVITY: ${WLNM_CHECK_CONNECTIVITY} (not used by direct k-fold partitioning)"
echo "WLNM_ADAPTIVE_CONNECTIVITY: ${WLNM_ADAPTIVE_CONNECTIVITY} (not used by direct k-fold partitioning)"
echo "WLNM_OUTPUT_ROOT: ${WLNM_OUTPUT_ROOT}"
echo "WLNM_MATLAB_JOB_STORAGE: ${WLNM_MATLAB_JOB_STORAGE}"
echo "Food web: ${foodweb_name}"
echo "Working directory: $(pwd)"

matlab -nodisplay -nosplash -batch "Main"

echo "Job finished at: $(date '+%Y-%m-%dT%H:%M:%S%z')"
