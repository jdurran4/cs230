#!/bin/bash
#SBATCH --job-name=M1.902_Re25.9_x0.18_y0.17_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
