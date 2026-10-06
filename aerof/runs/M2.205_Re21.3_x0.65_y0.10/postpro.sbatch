#!/bin/bash
#SBATCH --job-name=M2.205_Re21.3_x0.65_y0.10_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
