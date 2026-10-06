#!/bin/bash
#SBATCH --job-name=M1.565_Re65.2_x0.75_y0.37_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
