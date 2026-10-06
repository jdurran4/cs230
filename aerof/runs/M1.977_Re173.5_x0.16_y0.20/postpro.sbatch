#!/bin/bash
#SBATCH --job-name=M1.977_Re173.5_x0.16_y0.20_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
