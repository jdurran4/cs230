#!/bin/bash
#SBATCH --job-name=M2.032_Re52.8_x0.70_y0.06_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
