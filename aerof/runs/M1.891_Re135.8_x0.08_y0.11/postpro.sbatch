#!/bin/bash
#SBATCH --job-name=M1.891_Re135.8_x0.08_y0.11_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
