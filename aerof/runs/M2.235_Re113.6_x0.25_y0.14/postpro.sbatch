#!/bin/bash
#SBATCH --job-name=M2.235_Re113.6_x0.25_y0.14_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
