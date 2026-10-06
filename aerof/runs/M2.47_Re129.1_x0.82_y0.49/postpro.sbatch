#!/bin/bash
#SBATCH --job-name=M2.47_Re129.1_x0.82_y0.49_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
