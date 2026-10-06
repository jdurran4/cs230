#!/bin/bash
#SBATCH --job-name=M2.075_Re28.7_x0.33_y0.24_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
