#!/bin/bash
# Run this case's chain: steady/ (steady solve on the shared refined mesh), then the
# production (unsteady) run in this directory, restarted from the steady solution.
# The shared mesh stage must have finished first, with the same NP. A stage that already
# finished (DONE) is skipped, so rerunning after a production failure resumes there.
#   Cluster:      sbatch run.sbatch            (NP from --ntasks)
#   Workstation:  ./runChain.sh [NP]           (default 8)
cd "$(dirname "$0")"
NP=${SLURM_NTASKS:-${1:-8}}
MESH_DIR=../mesh                              # written by newRun.py

if [ ! -f "$MESH_DIR/DONE" ]; then
    echo "Error: the mesh stage has not finished. Run first:  $MESH_DIR/run.sh $NP"
    exit 1
fi
if [ -f steady/DONE ]; then
    echo "steady/ already done: skipping"
else
    steady/run.sh $NP || exit 1
fi
./run.sh $NP
