#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.%j.out
#SBATCH --error=test.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=mgardos
#SBATCH --partition=gpuq             # queue for job submission
#SBATCH --account=gpuq              # queue for job submission

srun ./test

python plot.py