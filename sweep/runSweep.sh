#!/bin/bash
# Workstation sweep: run every case in unsteady/runs.txt (written by sweep/newRun.py, in
# priority order) with its run.sh, JOBS cases at a time with NP MPI ranks each. A case whose DONE is newer than its FluidFile is
# skipped, so rerunning resumes the sweep. Postprocessing (runs/<case>/postpro.sh) is not
# run here, so the solver gets the machine; run it per case afterwards.
#   sweep/runSweep.sh                          every case in unsteady/runs.txt
#   sweep/runSweep.sh M1.5_Re100_x0.04_y0.12 ...   only these
#   JOBS=2 NP=8 sweep/runSweep.sh              defaults (frg-n05 has 20 cores)
# To keep it running after closing the terminal:
#   nohup sweep/runSweep.sh > ~/Research/cs230-data/unsteady/runSweep.log 2>&1 &
# Each case's AERO-F log is runs/<case>/log.out. To stop the sweep and its running cases,
# kill its process group:  kill -- -$(cat ~/Research/cs230-data/unsteady/runSweep.pgid)
cd "${CS230_DATA:-$HOME/Research/cs230-data}/unsteady"
ps -o pgid= -p $$ | tr -d ' ' > runSweep.pgid
NP=${NP:-8}
JOBS=${JOBS:-2}
if [ $# -gt 0 ]; then names=("$@"); else mapfile -t names < runs.txt; fi
cases=("${names[@]/#/runs/}")

runCase() {
    local dir=$1 np=$2
    if [ "$dir"/DONE -nt "$dir"/FluidFile ]; then
        echo "$(date '+%F %T') skip  $dir (already done)"
        return
    fi
    echo "$(date '+%F %T') start $dir"
    if "$dir"/run.sh "$np" > /dev/null 2>&1; then
        echo "$(date '+%F %T') done  $dir"
    else
        echo "$(date '+%F %T') FAIL  $dir (see $dir/log.out)"
    fi
}
export -f runCase
printf '%s\n' "${cases[@]}" | xargs -P "$JOBS" -I{} bash -c "runCase {} $NP"
