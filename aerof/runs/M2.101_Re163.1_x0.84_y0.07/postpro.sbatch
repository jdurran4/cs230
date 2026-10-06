#!/bin/bash
#SBATCH --job-name=M2.101_Re163.1_x0.84_y0.07_post
#SBATCH --time=2:00:00
#SBATCH --ntasks=12
#SBATCH --mem=16G

# Output frames are split across tasks
mpirun -np $SLURM_NTASKS bash postpro.sh |& tee log.postpro
