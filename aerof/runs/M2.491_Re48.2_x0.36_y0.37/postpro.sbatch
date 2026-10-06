#!/bin/bash
#SBATCH --job-name=M2.491_Re48.2_x0.36_y0.37_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
