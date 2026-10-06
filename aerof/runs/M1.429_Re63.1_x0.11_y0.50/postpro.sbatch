#!/bin/bash
#SBATCH --job-name=M1.429_Re63.1_x0.11_y0.50_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
