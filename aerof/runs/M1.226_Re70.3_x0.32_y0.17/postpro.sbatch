#!/bin/bash
#SBATCH --job-name=M1.226_Re70.3_x0.32_y0.17_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
