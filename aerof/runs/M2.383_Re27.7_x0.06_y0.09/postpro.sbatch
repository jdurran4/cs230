#!/bin/bash
#SBATCH --job-name=M2.383_Re27.7_x0.06_y0.09_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
