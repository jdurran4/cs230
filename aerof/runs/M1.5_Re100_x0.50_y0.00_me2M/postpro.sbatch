#!/bin/bash
#SBATCH --job-name=M1.5_Re100_x0.50_y0.00_me2M_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
