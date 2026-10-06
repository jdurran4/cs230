#!/bin/bash
#SBATCH --job-name=M1.434_Re108.5_x0.26_y0.03_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
