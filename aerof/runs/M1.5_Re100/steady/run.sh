#!/bin/bash
# Run AERO-F on this directory's FluidFile: one stage (steady or unsteady; see
# newRun.py). Does not clean: run ./clean.sh first to start over (it deletes restart
# files in references/). Writes DONE when the run finishes and left restart files.
#   Cluster:      sbatch run.sbatch            (NP from --ntasks)
#   Workstation:  ./run.sh [NP]                (default 8). To keep it running after
#                 closing the terminal:  nohup ./run.sh 8 > /dev/null 2>&1 &   (output in log.out)
# Both stages of a case must use the same NP: the unsteady stage reads the refined mesh
# from steady/references/, which has a CPU map only for the NP steady ran with (the
# steady stage reads the coarse mesh in data/: NP must be one of the -cpu counts in
# preprocess.sh).
cd "$(dirname "$0")"
ROOT=$PWD
until [ -f "$ROOT/tools.sh" ] || [ "$ROOT" = / ]; do ROOT=$(dirname "$ROOT"); done
source "$ROOT/tools.sh"
NP=${SLURM_NTASKS:-${1:-8}} # number of mpi processes

# The mesh this stage reads must have a CPU map for NP
GEOM=$(sed -n 's/^[[:space:]]*GeometryPrefix[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' FluidFile)
if [ ! -f "$GEOM.${NP}cpu" ]; then
    echo "Error: no CPU map $GEOM.${NP}cpu for NP = $NP. Available:" \
        $(cd "$(dirname "$GEOM")" 2>/dev/null && ls $(basename "$GEOM").*cpu 2>/dev/null)
    exit 1
fi

mkdir -p results references postpro
rm -f DONE

# run simulation
mpirun -np $NP $AEROF FluidFile |& tee log.out
STATUS=${PIPESTATUS[0]}
if [ $STATUS -eq 0 ] && [ -f references/MeshData ]; then
    touch DONE
else
    echo "Run failed (exit status $STATUS) or wrote no restart files: see log.out"
    exit 1
fi
