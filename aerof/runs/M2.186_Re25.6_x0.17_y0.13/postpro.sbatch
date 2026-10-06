#!/bin/bash
#SBATCH --job-name=M2.186_Re25.6_x0.17_y0.13_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
