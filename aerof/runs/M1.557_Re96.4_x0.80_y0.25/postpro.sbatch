#!/bin/bash
#SBATCH --job-name=M1.557_Re96.4_x0.80_y0.25_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
