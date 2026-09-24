#!/bin/bash
# Run AERO-F for this case. Does not clean: run ./clean.sh first to start over
# (it deletes restart files in references/).
cd "$(dirname "$0")"
source ../../tools.sh
NP=${SLURM_NTASKS:-48} # number of mpi processes (sbatch --ntasks, else 48)

mkdir -p results references postpro

# run simulation
# module load cmake/3.8.1 gcc/9.1.0 openmpi/4.1.2 imkl/2019
mpirun -np $NP $AEROF FluidFile |& tee log.out
