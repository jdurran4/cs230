#!/bin/bash
#SBATCH --job-name=M2.303_Re161.9_x0.83_y0.26_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
