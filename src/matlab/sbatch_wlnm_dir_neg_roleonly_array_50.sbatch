#!/bin/bash
#SBATCH --job-name=WLNM_DN_ROLE
#SBATCH --chdir=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework-dir-neg/src/matlab
#SBATCH --output=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework-dir-neg/slurm_logs/%x.%A_%a.out
#SBATCH --error=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework-dir-neg/slurm_logs/%x.%A_%a.err
#SBATCH --array=1-290%4
#SBATCH --nodes=1
#SBATCH --ntasks=51
#SBATCH --partition=highmem
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acw792@qmul.ac.uk

set -euo pipefail

: "${WLNM_CONDITION_ID:?Submit this template through launch_wlnm_dir_neg_roleonly.sh}"
: "${WLNM_OUTPUT_ROOT:?Missing isolated output root}"
: "${WLNM_NUM_EXPERIMENTS:?Missing number of experiments}"
: "${WLNM_PARALLEL_WORKERS:?Missing parallel worker count}"
: "${WLNM_TRAIN_RATIO_RANGE:?Missing train-ratio range}"

module purge
module load matlab

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WLNM_MATLAB_JOB_STORAGE="${TMPDIR:-/tmp}/wlnm_matlab_jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WLNM_MATLAB_JOB_STORAGE"

export WLNM_VERSION=WLNM_dir_neg
export WLNM_CV_ENABLED=false
export WLNM_SWEEP_TRAIN_RATIOS=true
export WLNM_THRESHOLD_MODE=fixed
export WLNM_FIXED_THRESHOLD=0.50
export WLNM_THRESHOLD_SWEEP_ENABLED=false
export WLNM_NEGATIVE_ELIGIBILITY_MODE=role_only
export WLNM_NEGATIVE_POSITIVE_RATIO=2
export WLNM_NEGATIVE_SAMPLING_STRATEGY=uniform_without_replacement
export WLNM_NEGATIVE_TOPUP_POLICY=uniform_remaining_nonlinks
export WLNM_NEGATIVE_MASS_ELIGIBILITY_ENABLED=false
export WLNM_NEGATIVE_MASS_ELIGIBILITY_THRESHOLD=1.0
export WLNM_CHECK_CONNECTIVITY=false
export WLNM_ADAPTIVE_CONNECTIVITY=false
export WLNM_BASE_SEED=12345
export WLNM_RESAMPLE_SPLITS_EACH_EXPERIMENT=true
export WLNM_FOODWEB_INDEX="${SLURM_ARRAY_TASK_ID}"
export WLNM_FOODWEB_CSV=data/foodwebs_mat/foodweb_metrics_ecosystem.csv
export WLNM_EXPORT_AUXILIARY_CSVS=false
export WLNM_CV_SAVE_CONFUSION=false
export WLNM_RUN_DELTA_TTESTS=false
export WLNM_RUN_DELTA_EQUIVALENCE=false
export WLNM_COMPUTE_ECOLOGICAL_METRICS=true

foodweb_name=$(python3 -c '
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
index = int(sys.argv[2]) - 1
if 0 <= index < len(rows):
    print(rows[index]["Foodweb"])
' "$WLNM_FOODWEB_CSV" "$SLURM_ARRAY_TASK_ID")

if [[ -z "$foodweb_name" ]]; then
    echo "ERROR: Could not resolve food web for array task ${SLURM_ARRAY_TASK_ID}." >&2
    exit 64
fi

result_csv="${WLNM_OUTPUT_ROOT}/prediction_scores_logs/${foodweb_name}_results_random_wlnm_dir_neg.csv"
terminal_log="${WLNM_OUTPUT_ROOT}/terminal_logs/${foodweb_name}_wlnm_dir_neg_terminal_log.txt"
completion_marker="${WLNM_OUTPUT_ROOT}/completion_markers/${foodweb_name}.complete"

if [[ -e "$result_csv" || -e "$terminal_log" || -e "$completion_marker" ]]; then
    echo "ERROR: Output collision for ${foodweb_name}." >&2
    echo "Existing result: ${result_csv}" >&2
    echo "Existing terminal log: ${terminal_log}" >&2
    echo "Existing completion marker: ${completion_marker}" >&2
    echo "Use an isolated retry output root instead of appending to this run." >&2
    exit 73
fi

echo "Job started at: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "Running on node: $(hostname)"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "WLNM_CONDITION_ID: ${WLNM_CONDITION_ID}"
echo "WLNM_TRAIN_RATIO_RANGE: ${WLNM_TRAIN_RATIO_RANGE}"
echo "WLNM_NEGATIVE_ELIGIBILITY_MODE: ${WLNM_NEGATIVE_ELIGIBILITY_MODE}"
echo "WLNM_NEGATIVE_POSITIVE_RATIO: ${WLNM_NEGATIVE_POSITIVE_RATIO}"
echo "WLNM_NEGATIVE_SAMPLING_STRATEGY: ${WLNM_NEGATIVE_SAMPLING_STRATEGY}"
echo "WLNM_NEGATIVE_TOPUP_POLICY: ${WLNM_NEGATIVE_TOPUP_POLICY}"
echo "WLNM_CHECK_CONNECTIVITY: ${WLNM_CHECK_CONNECTIVITY}"
echo "WLNM_ADAPTIVE_CONNECTIVITY: ${WLNM_ADAPTIVE_CONNECTIVITY}"
echo "WLNM_OUTPUT_ROOT: ${WLNM_OUTPUT_ROOT}"
echo "WLNM_MATLAB_JOB_STORAGE: ${WLNM_MATLAB_JOB_STORAGE}"
echo "Food web: ${foodweb_name}"
echo "Working directory: $(pwd)"

matlab -nodisplay -nosplash -batch "Main"

{
    echo "Foodweb=${foodweb_name}"
    echo "ArrayTaskID=${SLURM_ARRAY_TASK_ID}"
    echo "JobID=${SLURM_JOB_ID}"
    echo "CompletedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "$completion_marker"

echo "Job finished at: $(date '+%Y-%m-%dT%H:%M:%S%z')"
