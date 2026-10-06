#!/bin/bash
#SBATCH --job-name=M1.607_Re68.5_x0.95_y0.01_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
