#!/bin/bash
#SBATCH --job-name=M1.852_Re187.6_x0.22_y0.36_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
