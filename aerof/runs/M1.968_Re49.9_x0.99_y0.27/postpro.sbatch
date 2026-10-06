#!/bin/bash
#SBATCH --job-name=M1.968_Re49.9_x0.99_y0.27_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
