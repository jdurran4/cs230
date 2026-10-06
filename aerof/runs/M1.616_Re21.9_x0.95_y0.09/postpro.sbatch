#!/bin/bash
#SBATCH --job-name=M1.616_Re21.9_x0.95_y0.09_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
