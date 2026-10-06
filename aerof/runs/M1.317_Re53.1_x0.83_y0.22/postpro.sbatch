#!/bin/bash
#SBATCH --job-name=M1.317_Re53.1_x0.83_y0.22_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
