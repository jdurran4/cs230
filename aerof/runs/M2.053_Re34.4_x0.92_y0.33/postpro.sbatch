#!/bin/bash
#SBATCH --job-name=M2.053_Re34.4_x0.92_y0.33_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
