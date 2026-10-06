#!/bin/bash
#SBATCH --job-name=M1.778_Re80.1_x0.11_y0.31_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
