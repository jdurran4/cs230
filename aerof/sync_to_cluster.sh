#!/bin/bash
# Sync the AERO-F setup and run directories to your computing cluster.
#
# Usage:
#   ./sync_to_cluster.sh                    # shared files + every case in runs/
#   ./sync_to_cluster.sh M1.5_Re100 M2_Re100  # shared files + only these cases
#
# The cluster gets the same layout as aerof/ here, since each FluidFile points at
# ../../data, ../../sources and ../../tools.sh:
#   REMOTE_BASE/tools.sh, preprocess.sh, sources/*.top
#   REMOTE_BASE/runs/<case>/FluidFile, *.sh, *.sbatch
# Only inputs are sent. Outputs on the cluster (data/, results/, references/,
# postpro/, logs) are never touched, so run again anytime to update inputs.

# ── Configure once ────────────────────────────────────────────────────────────
HOST=independence2
REMOTE_BASE=/home/tdurrant/cs230/aerof

# ── Args ──────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"
if [ $# -gt 0 ]; then
    CASES=("$@")
else
    CASES=($(cd runs && ls -d */ | tr -d /))
fi

for c in "${CASES[@]}"; do
    if [ ! -f "runs/$c/FluidFile" ]; then
        echo "Error: runs/$c/FluidFile does not exist. Run newRun.py first."
        exit 1
    fi
done
if [ ! -f sources/domain.top ] || [ ! -f sources/embeddedSurface.top ]; then
    echo "Error: sources/*.top missing. Run cylMeshGen.py first."
    exit 1
fi

# One rsync call = one connection = one password/2FA prompt. -R keeps the paths
# relative to aerof/, and the remote mkdir creates REMOTE_BASE on first sync.
echo "Syncing shared files + ${CASES[*]} → $HOST:$REMOTE_BASE ..."
rsync -avzR --rsync-path="mkdir -p $REMOTE_BASE && rsync" \
    --exclude=results/ --exclude=references/ --exclude=postpro/ \
    --exclude="log.*" --exclude="*.exo" \
    tools.sh preprocess.sh sources/domain.top sources/embeddedSurface.top \
    "${CASES[@]/#/runs/}" "$HOST:$REMOTE_BASE/" || exit 1

echo ""
echo "Done. First time only (or after changing the mesh), preprocess on the cluster:"
echo "  ssh $HOST 'cd $REMOTE_BASE && ./preprocess.sh'"
echo "To submit a case:"
echo "  ssh $HOST 'cd $REMOTE_BASE/runs/${CASES[0]} && sbatch run.sbatch'"
