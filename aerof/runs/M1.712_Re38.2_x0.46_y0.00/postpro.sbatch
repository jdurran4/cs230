#!/bin/bash
#SBATCH --job-name=M1.712_Re38.2_x0.46_y0.00_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
