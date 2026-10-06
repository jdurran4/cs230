#!/bin/bash
#SBATCH --job-name=M1.337_Re51.2_x0.34_y0.19_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
