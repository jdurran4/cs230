#!/bin/bash
#SBATCH --job-name=M1.765_Re53.8_x0.71_y0.22_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
