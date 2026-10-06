#!/bin/bash
#SBATCH --job-name=M2.223_Re108.9_x0.28_y0.41_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
