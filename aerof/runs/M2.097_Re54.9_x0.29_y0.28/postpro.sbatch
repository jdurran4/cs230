#!/bin/bash
#SBATCH --job-name=M2.097_Re54.9_x0.29_y0.28_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
