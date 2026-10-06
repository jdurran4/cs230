#!/bin/bash
#SBATCH --job-name=M1.941_Re46.2_x0.48_y0.17_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
