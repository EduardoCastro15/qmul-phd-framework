#!/bin/bash
#SBATCH --job-name=WLNM_DN_FIX4
#SBATCH --chdir=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework/src/matlab
#SBATCH --output=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework/slurm_logs/%x.%j.out
#SBATCH --error=/data/home/acw792/PhD/Stage2/Code/qmul-phd-framework/slurm_logs/%x.%j.err

#SBATCH --ntasks=20
#SBATCH --partition=highmem
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acw792@qmul.ac.uk

set -euo pipefail

module purge
module load matlab

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "Working directory: $(pwd)"

matlab -nodisplay -nosplash -batch "Main"

echo "Job finished at: $(date)"
