#!/bin/bash
#SBATCH --job-name=WLNM_YUJIE_OZ
# Working directory, array bounds, resources, and logs come from the launcher.
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
: "${WLNM_FOODWEB_CSV:?Missing input network manifest}"
: "${WLNM_MAT_FOLDER:?Missing preprocessed MAT folder}"
: "${SLURM_ARRAY_TASK_ID:?Missing array task ID}"

module purge
module load matlab

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WLNM_MATLAB_JOB_STORAGE="${TMPDIR:-/tmp}/wlnm_yujie_jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WLNM_MATLAB_JOB_STORAGE"

export WLNM_VERSION=WLNM_dir_neg
export WLNM_USE_PARALLEL=true
export WLNM_USE_GRAPH_ENCODING_PARALLEL=false
export WLNM_K_RANGE=10
export WLNM_CV_ENABLED=false
export WLNM_USE_BACKBONE=false
export WLNM_THRESHOLD_MODE=fixed
export WLNM_FIXED_THRESHOLD=0.50
export WLNM_THRESHOLD_SWEEP_ENABLED=false
export WLNM_NEGATIVE_ELIGIBILITY_MODE=observed_zero
export WLNM_NEGATIVE_POSITIVE_RATIO=2
export WLNM_NEGATIVE_SAMPLING_STRATEGY=uniform_without_replacement
export WLNM_NEGATIVE_TOPUP_POLICY=error
export WLNM_NEGATIVE_MASS_ELIGIBILITY_ENABLED=false
export WLNM_CHECK_CONNECTIVITY=false
export WLNM_ADAPTIVE_CONNECTIVITY=false
export WLNM_BASE_SEED=12345
export WLNM_RESAMPLE_SPLITS_EACH_EXPERIMENT=true
export WLNM_FOODWEB_INDEX="${SLURM_ARRAY_TASK_ID}"
export WLNM_EXPORT_AUXILIARY_CSVS=false
export WLNM_CV_SAVE_CONFUSION=false
export WLNM_RUN_DELTA_TTESTS=false
export WLNM_RUN_DELTA_EQUIVALENCE=false
export WLNM_COMPUTE_ECOLOGICAL_METRICS=false

foodweb_count=$(python3 -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
' "$WLNM_FOODWEB_CSV")
if [[ "$SLURM_ARRAY_TASK_ID" -lt 1 || "$SLURM_ARRAY_TASK_ID" -gt "$foodweb_count" ]]; then
    echo "ERROR: Food-web index outside 1-${foodweb_count}: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 65
fi

foodweb_name=$(python3 -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
print(rows[int(sys.argv[2]) - 1]["Foodweb"])
' "$WLNM_FOODWEB_CSV" "$SLURM_ARRAY_TASK_ID")
expected_mat_hash=$(python3 -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
print(rows[int(sys.argv[2]) - 1]["mat_sha256"])
' "$WLNM_FOODWEB_CSV" "$SLURM_ARRAY_TASK_ID")

input_mat="${WLNM_MAT_FOLDER}/${foodweb_name}.mat"
result_csv="${WLNM_OUTPUT_ROOT}/prediction_scores_logs/${foodweb_name}_results_random_wlnm_dir_neg.csv"
terminal_log="${WLNM_OUTPUT_ROOT}/terminal_logs/${foodweb_name}_wlnm_dir_neg_terminal_log.txt"
completion_marker="${WLNM_OUTPUT_ROOT}/completion_markers/${foodweb_name}.complete"

if [[ ! -f "$input_mat" ]]; then
    echo "ERROR: Missing preprocessed MAT: ${input_mat}" >&2
    exit 66
fi
actual_mat_hash=$(python3 -c '
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(block)
print(h.hexdigest())
' "$input_mat")
if [[ "$actual_mat_hash" != "$expected_mat_hash" ]]; then
    echo "ERROR: Preprocessed MAT SHA-256 mismatch for ${foodweb_name}." >&2
    exit 74
fi
if [[ -e "$result_csv" || -e "$terminal_log" || -e "$completion_marker" ]]; then
    echo "ERROR: Output collision for ${foodweb_name}." >&2
    exit 73
fi

echo "Job started: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "Food web: ${foodweb_name}"
echo "Experiments: ${WLNM_NUM_EXPERIMENTS}"
echo "Input MAT: ${input_mat}"
echo "Output root: ${WLNM_OUTPUT_ROOT}"

matlab -nodisplay -nosplash -batch "Main"

marker_tmp="${completion_marker}.tmp.${SLURM_JOB_ID}"
{
    echo "Foodweb=${foodweb_name}"
    echo "ArrayTaskID=${SLURM_ARRAY_TASK_ID}"
    echo "JobID=${SLURM_JOB_ID}"
    echo "InputMatSHA256=${actual_mat_hash}"
    echo "CompletedAt=$(date '+%Y-%m-%dT%H:%M:%S%z')"
} > "$marker_tmp"
mv "$marker_tmp" "$completion_marker"

echo "Job finished: $(date '+%Y-%m-%dT%H:%M:%S%z')"
