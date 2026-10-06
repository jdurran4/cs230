#!/bin/bash
#SBATCH --job-name=M2_Re20.4_x0.66_y0.14
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=16G
module load icc/2018 ifort/2018
module swap openmpi/3.0.0-gcc-7.1.0 openmpi/3.0.0-intel-2018.u1

./run.sh
