#!/bin/bash
#SBATCH --job-name=M2.193_Re84.2_x0.04_y0.04_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
