#!/bin/bash
#SBATCH --job-name=M1.284_Re150.5_x0.09_y0.34_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
