#!/bin/bash
#SBATCH --job-name=M1.329_Re32.8_x0.31_y0.14_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
