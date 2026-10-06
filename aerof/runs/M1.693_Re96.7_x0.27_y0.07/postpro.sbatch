#!/bin/bash
#SBATCH --job-name=M1.693_Re96.7_x0.27_y0.07_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
