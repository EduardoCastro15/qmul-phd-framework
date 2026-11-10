#!/bin/bash
#SBATCH --job-name=WLNM_FMASS
#SBATCH --array=1-290%4
#SBATCH --nodes=1
#SBATCH --ntasks=101
#SBATCH --partition=highmem
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acw792@qmul.ac.uk

set -euo pipefail

: "${WLNM_OUTPUT_ROOT:?Missing isolated variant output root}"
: "${WLNM_FOCAL_MASS_FEATURE_MODE:?Missing focal-mass ablation mode}"
: "${WLNM_NUM_EXPERIMENTS:?Missing experiment count}"
: "${WLNM_PARALLEL_WORKERS:?Missing worker count}"
: "${WLNM_TRAIN_RATIO_RANGE:?Missing train-ratio range}"
: "${WLNM_FOODWEB_CSV:?Missing food-web CSV}"

module purge
module load matlab

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WLNM_MATLAB_JOB_STORAGE="${TMPDIR:-/tmp}/wlnm_focal_mass/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WLNM_MATLAB_JOB_STORAGE"

export WLNM_VERSION=WLNM_dir_neg_focal_mass
export WLNM_USE_PARALLEL=true
export WLNM_CV_ENABLED=false
export WLNM_K_RANGE=10
export WLNM_SWEEP_TRAIN_RATIOS=true
export WLNM_THRESHOLD_MODE=fixed
export WLNM_FIXED_THRESHOLD=0.50
export WLNM_THRESHOLD_SWEEP_ENABLED=false
export WLNM_NEGATIVE_ELIGIBILITY_MODE=role_only
export WLNM_NEGATIVE_POSITIVE_RATIO=2
export WLNM_NEGATIVE_SAMPLING_STRATEGY=uniform_without_replacement
export WLNM_NEGATIVE_TOPUP_POLICY=uniform_remaining_nonlinks
export WLNM_NEGATIVE_MASS_ELIGIBILITY_ENABLED=false
export WLNM_TROPHIC_LEVEL_PROTOCOL=validated_v2
export WLNM_TROPHIC_HIGH_PRECISION=auto
export WLNM_CHECK_CONNECTIVITY=false
export WLNM_ADAPTIVE_CONNECTIVITY=false
export WLNM_BASE_SEED=12345
export WLNM_RESAMPLE_SPLITS_EACH_EXPERIMENT=true
export WLNM_FOODWEB_INDEX="${SLURM_ARRAY_TASK_ID}"
export WLNM_EXPORT_AUXILIARY_CSVS=false
export WLNM_CV_SAVE_CONFUSION=false
export WLNM_RUN_DELTA_TTESTS=false
export WLNM_RUN_DELTA_EQUIVALENCE=false
export WLNM_COMPUTE_ECOLOGICAL_METRICS=true

foodweb_name=$(python3 -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
index = int(sys.argv[2]) - 1
if 0 <= index < len(rows):
    print(rows[index]["Foodweb"])
' "$WLNM_FOODWEB_CSV" "$SLURM_ARRAY_TASK_ID")

if [[ -z "$foodweb_name" ]]; then
    echo "ERROR: Could not resolve food web for task ${SLURM_ARRAY_TASK_ID}." >&2
    exit 64
fi

result_csv="${WLNM_OUTPUT_ROOT}/prediction_scores_logs/${foodweb_name}_results_random_wlnm_dir_neg_focal_mass.csv"
terminal_log="${WLNM_OUTPUT_ROOT}/terminal_logs/${foodweb_name}_wlnm_dir_neg_focal_mass_terminal_log.txt"
completion_marker="${WLNM_OUTPUT_ROOT}/completion_markers/${foodweb_name}.complete"

if [[ -e "$result_csv" || -e "$terminal_log" || -e "$completion_marker" ]]; then
    echo "ERROR: Refusing to append or overwrite outputs for ${foodweb_name}." >&2
    exit 73
fi

echo "Job started at: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "Version: ${WLNM_VERSION}"
echo "Focal mass mode: ${WLNM_FOCAL_MASS_FEATURE_MODE}"
echo "Food web: ${foodweb_name}"
echo "Output root: ${WLNM_OUTPUT_ROOT}"

matlab -nodisplay -nosplash -batch "Main"

{
    echo "Foodweb=${foodweb_name}"
    echo "FocalMassFeatureMode=${WLNM_FOCAL_MASS_FEATURE_MODE}"
    echo "ArrayTaskID=${SLURM_ARRAY_TASK_ID}"
    echo "JobID=${SLURM_JOB_ID}"
    echo "CompletedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "$completion_marker"

echo "Job finished at: $(date '+%Y-%m-%dT%H:%M:%S%z')"
