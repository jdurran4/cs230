#!/bin/bash
#SBATCH --job-name=M1.254_Re116.4_x0.32_y0.42_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
