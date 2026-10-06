#!/bin/bash
# Wait for runSweep.sh to finish, then multi-grid bin every finished run that has no
# samples.npz yet (binMulti.py, JOBS at a time) and merge into dataset_multi.npz.
#   nohup ./postSweep.sh > postSweep.log 2>&1 &
cd "$(dirname "$0")"
JOBS=${JOBS:-12}
while pgrep -f runSweep.sh > /dev/null; do sleep 600; done
echo "$(date '+%F %T') sweep finished: $(grep -c 'done ' runSweep.log) done, $(grep -c FAIL runSweep.log) FAIL"
../.venv/bin/python -c "import binMulti as b, os
print('\n'.join(r for r in b.finishedRuns() if not os.path.isfile(os.path.join(b.runsDir, r, 'samples.npz'))))" |
    xargs -P "$JOBS" -I{} sh -c '../.venv/bin/python binMulti.py {} > binlogs/multi_{}.log 2>&1 || echo "FAIL bin {}"'
echo "$(date '+%F %T') binning finished"
../.venv/bin/python binMulti.py merge
