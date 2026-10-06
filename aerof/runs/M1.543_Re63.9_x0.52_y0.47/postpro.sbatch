#!/bin/bash
#SBATCH --job-name=M1.543_Re63.9_x0.52_y0.47_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
