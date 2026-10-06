#!/bin/bash
#SBATCH --job-name=M1.727_Re23.8_x0.49_y0.02_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
