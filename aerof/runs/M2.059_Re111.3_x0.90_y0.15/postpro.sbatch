#!/bin/bash
#SBATCH --job-name=M2.059_Re111.3_x0.90_y0.15_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
