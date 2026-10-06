#!/bin/bash
#SBATCH --job-name=M2.344_Re29.5_x0.91_y0.46_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
