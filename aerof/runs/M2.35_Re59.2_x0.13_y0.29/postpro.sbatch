#!/bin/bash
#SBATCH --job-name=M2.35_Re59.2_x0.13_y0.29_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
