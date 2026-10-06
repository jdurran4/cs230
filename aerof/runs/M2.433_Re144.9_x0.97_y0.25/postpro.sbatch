#!/bin/bash
#SBATCH --job-name=M2.433_Re144.9_x0.97_y0.25_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
