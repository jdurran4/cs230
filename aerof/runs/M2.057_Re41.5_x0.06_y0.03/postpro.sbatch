#!/bin/bash
#SBATCH --job-name=M2.057_Re41.5_x0.06_y0.03_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
