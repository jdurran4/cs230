#!/bin/bash
#SBATCH --job-name=M1.736_Re24.1_x0.05_y0.47_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
