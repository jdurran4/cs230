#!/bin/bash
#SBATCH --job-name=M2.038_Re78.4_x0.98_y0.06_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
