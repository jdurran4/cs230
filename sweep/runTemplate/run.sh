#!/bin/bash
# Run AERO-F on this case's FluidFile (unsteady; see sweep/newRun.py). Does not clean: run
# ./clean.sh first to start over (it deletes restart files in references/). Writes DONE
# when the run finishes without aborting and left restart files.
#   Cluster:      sbatch run.sbatch            (NP from --ntasks)
#   Workstation:  ./run.sh [NP]                (default 8). To keep it running after
#                 closing the terminal:  nohup ./run.sh 8 > /dev/null 2>&1 &   (output in log.out)
# NP must be one of the -cpu counts in sweep/preprocess.sh (the run reads the coarse mesh
# in unsteady/data/).
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
# AERO-F exits 0 even when it aborts ("Cannot further reduce CFL number. Aborting.")
if [ $STATUS -eq 0 ] && [ -f references/MeshData ] && ! grep -q 'Aborting' log.out; then
    touch DONE
else
    echo "Run failed (exit status $STATUS) or wrote no restart files: see log.out"
    exit 1
fi
