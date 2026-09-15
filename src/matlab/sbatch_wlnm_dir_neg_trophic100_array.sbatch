#!/bin/bash
#SBATCH --job-name=WLNM_DN_TL100
# Working directory and log paths are supplied by the launcher.
#SBATCH --array=210,215%2
#SBATCH --nodes=1
#SBATCH --ntasks=51
#SBATCH --partition=highmem
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acw792@qmul.ac.uk

set -euo pipefail

: "${WLNM_OUTPUT_ROOT:?Missing isolated output root}"
: "${WLNM_NUM_EXPERIMENTS:?Missing number of experiments}"
: "${WLNM_PARALLEL_WORKERS:?Missing parallel worker count}"

module purge
module load matlab

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WLNM_MATLAB_JOB_STORAGE="${TMPDIR:-/tmp}/wlnm_matlab_jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WLNM_MATLAB_JOB_STORAGE"

export WLNM_VERSION=WLNM_dir_neg
export WLNM_USE_PARALLEL=true
export WLNM_CV_ENABLED=false
export WLNM_SWEEP_TRAIN_RATIOS=true
export WLNM_TRAIN_RATIO_RANGE="${WLNM_TRAIN_RATIO_RANGE:-0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90}"
export WLNM_THRESHOLD_MODE=fixed
export WLNM_FIXED_THRESHOLD=0.50
export WLNM_THRESHOLD_SWEEP_ENABLED=false
export WLNM_NEGATIVE_ELIGIBILITY_MODE=role_only
export WLNM_NEGATIVE_POSITIVE_RATIO=2
export WLNM_NEGATIVE_SAMPLING_STRATEGY=uniform_without_replacement
export WLNM_NEGATIVE_TOPUP_POLICY=uniform_remaining_nonlinks
export WLNM_NEGATIVE_MASS_ELIGIBILITY_ENABLED=false
export WLNM_NEGATIVE_MASS_ELIGIBILITY_THRESHOLD=1.0
export WLNM_TROPHIC_LEVEL_PROTOCOL=validated_v2
export WLNM_TROPHIC_HIGH_PRECISION=auto
export WLNM_TROPHIC_SNAPSHOT_DIR="${WLNM_OUTPUT_ROOT}/ecological_snapshots"
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
print(rows[int(sys.argv[2]) - 1]["Foodweb"])
' "$WLNM_FOODWEB_CSV" "$SLURM_ARRAY_TASK_ID")

if [[ "$SLURM_ARRAY_TASK_ID" -lt 1 || "$SLURM_ARRAY_TASK_ID" -gt 290 ]]; then
    echo "ERROR: Food-web index outside 1-290: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 65
fi

result_csv="${WLNM_OUTPUT_ROOT}/prediction_scores_logs/${foodweb_name}_results_random_wlnm_dir_neg.csv"
terminal_log="${WLNM_OUTPUT_ROOT}/terminal_logs/${foodweb_name}_wlnm_dir_neg_terminal_log.txt"
completion_marker="${WLNM_OUTPUT_ROOT}/completion_markers/${foodweb_name}.complete"

if [[ -e "$result_csv" || -e "$terminal_log" || -e "$completion_marker" ]]; then
    echo "ERROR: Output collision for ${foodweb_name}." >&2
    exit 73
fi

echo "Job started: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "Food web: ${foodweb_name}"
echo "Experiments: ${WLNM_NUM_EXPERIMENTS}"
echo "Output root: ${WLNM_OUTPUT_ROOT}"

matlab -nodisplay -nosplash -batch "Main"

{
    echo "Foodweb=${foodweb_name}"
    echo "ArrayTaskID=${SLURM_ARRAY_TASK_ID}"
    echo "JobID=${SLURM_JOB_ID}"
    echo "CompletedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "$completion_marker"

echo "Job finished: $(date '+%Y-%m-%dT%H:%M:%S%z')"
