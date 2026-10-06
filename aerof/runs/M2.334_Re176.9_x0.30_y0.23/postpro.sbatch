#!/bin/bash
#SBATCH --job-name=M2.334_Re176.9_x0.30_y0.23_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
