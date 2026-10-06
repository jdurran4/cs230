#!/bin/bash
#SBATCH --job-name=M1.272_Re38.6_x0.12_y0.05_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
