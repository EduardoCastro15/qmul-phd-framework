#!/bin/bash
#SBATCH --job-name=WLNM_DIR_NEG_MAIN
#SBATCH --chdir=/home/acw792/PhD/Stage2/Code/qmul-phd-framework/src/matlab
#SBATCH --output=/home/acw792/PhD/Stage2/Code/qmul-phd-framework/slurm_logs/%x.%j.out
#SBATCH --error=/home/acw792/PhD/Stage2/Code/qmul-phd-framework/slurm_logs/%x.%j.err

#SBATCH --ntasks=20
#SBATCH --partition=highmem
#SBATCH --mem-per-cpu=16G
#SBATCH --time=10:00:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acw792@qmul.ac.uk

module purge
module load matlab

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "Working directory: $(pwd)"

matlab -nodisplay -batch "Main"

echo "Job finished at: $(date)"