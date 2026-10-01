#!/bin/bash
# Run this case's chain: steady/ (pre-refines the coarse mesh, then solves steady), then
# the production (unsteady) run in this directory, restarted from the steady solution on
# its refined mesh, with the same NP. A stage that already finished (DONE) is skipped,
# so rerunning after a production failure resumes there.
#   Cluster:      sbatch run.sbatch            (NP from --ntasks)
#   Workstation:  ./runChain.sh [NP]           (default 8)
cd "$(dirname "$0")"
NP=${SLURM_NTASKS:-${1:-8}}

if [ -f steady/DONE ]; then
    echo "steady/ already done: skipping"
else
    steady/run.sh $NP || exit 1
fi
./run.sh $NP
