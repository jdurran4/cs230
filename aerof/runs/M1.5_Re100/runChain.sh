#!/bin/bash
# Run this case's chain: steady/ (pre-refines the coarse mesh, then solves steady), then
# the production (unsteady) run in this directory, restarted from the steady solution on
# its refined mesh, with the same NP. A steady stage that already finished (DONE newer
# than its FluidFile) is skipped, so rerunning after a production failure resumes there;
# regenerating the case with newRun.py makes the steady stage run again.
#   Cluster:      sbatch run.sbatch            (NP from --ntasks)
#   Workstation:  ./runChain.sh [NP]           (default 8)
cd "$(dirname "$0")"
NP=${SLURM_NTASKS:-${1:-8}}
RUN_UNSTEADY=0                                 # written by newRun.py (unsteadyCases)

if [ steady/DONE -nt steady/FluidFile ]; then
    echo "steady/ already done: skipping"
else
    steady/run.sh $NP || exit 1
fi
if [ $RUN_UNSTEADY -eq 1 ]; then
    ./run.sh $NP
else
    echo "Steady-only case: stopping after steady/ (./run.sh $NP runs the production stage)"
fi
