#!/bin/bash
# Run AERO-F for this case. Does not clean: run ./clean.sh first to start over
# (it deletes restart files in references/).
#   Cluster:      sbatch run.sbatch            (NP from --ntasks)
#   Workstation:  ./run.sh [NP]                (default 8; NP must be one of the
#                 -cpu counts in preprocess.sh). To keep it running after closing
#                 the terminal:  nohup ./run.sh 8 > /dev/null 2>&1 &   (output in log.out)
cd "$(dirname "$0")"
source ../../tools.sh
NP=${SLURM_NTASKS:-${1:-8}} # number of mpi processes

mkdir -p results references postpro

# run simulation
mpirun -np $NP $AEROF FluidFile |& tee log.out
