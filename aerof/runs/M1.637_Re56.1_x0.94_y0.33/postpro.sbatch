#!/bin/bash
#SBATCH --job-name=M1.637_Re56.1_x0.94_y0.33_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
