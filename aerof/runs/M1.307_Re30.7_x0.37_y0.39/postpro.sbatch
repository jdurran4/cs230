#!/bin/bash
#SBATCH --job-name=M1.307_Re30.7_x0.37_y0.39_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
