#!/bin/bash
#SBATCH --job-name=WLNM_DN_KF20
#SBATCH --chdir=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework-dir-neg-kfold/src/matlab
#SBATCH --output=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework-dir-neg-kfold/slurm_logs/%x.%A_%a.out
#SBATCH --error=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework-dir-neg-kfold/slurm_logs/%x.%A_%a.err
#SBATCH --array=1-290%2
#SBATCH --nodes=1
#SBATCH --ntasks=21
#SBATCH --partition=highmem
#SBATCH --mem-per-cpu=7G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acw792@qmul.ac.uk

set -euo pipefail

module purge
module load matlab

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WLNM_MATLAB_JOB_STORAGE="${TMPDIR:-/tmp}/wlnm_matlab_jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WLNM_MATLAB_JOB_STORAGE"

export WLNM_VERSION=WLNM_dir_neg_kfold
export WLNM_CV_ENABLED=true
export WLNM_CV_K_LIST=3,5,10
export WLNM_NUM_EXPERIMENTS=20
export WLNM_PARALLEL_WORKERS=20
export WLNM_FOODWEB_INDEX="${SLURM_ARRAY_TASK_ID}"
export WLNM_FOODWEB_CSV=data/foodwebs_mat/foodweb_metrics_ecosystem.csv
export WLNM_OUTPUT_ROOT=data/result_wlnm_dir_neg_kfold_role_constraints_hybrid_20x290_Apocrita
export WLNM_EXPORT_AUXILIARY_CSVS=false
export WLNM_CV_SAVE_CONFUSION=false
export WLNM_RUN_DELTA_TTESTS=false
export WLNM_RUN_DELTA_EQUIVALENCE=false
export WLNM_COMPUTE_ECOLOGICAL_METRICS=true
export WLNM_THRESHOLD_SWEEP_ENABLED=true
export WLNM_THRESHOLD_SWEEP_RANGE=0.10:0.10:0.90
export WLNM_NEGATIVE_MASS_PREFERENCE_ENABLED=true
export WLNM_NEGATIVE_MASS_PREFERENCE_THRESHOLD=1.0

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "WLNM_MATLAB_JOB_STORAGE: ${WLNM_MATLAB_JOB_STORAGE}"
echo "Working directory: $(pwd)"

matlab -nodisplay -nosplash -batch "Main"

echo "Job finished at: $(date)"
