#!/bin/bash
#SBATCH --job-name=M2.398_Re71.3_x0.76_y0.30_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
