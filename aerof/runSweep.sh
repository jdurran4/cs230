#!/bin/bash
# Workstation sweep: run every case (run.sh) and convert its output for binInputs
# (postpro.sh), JOBS cases at a time with NP MPI ranks each. A case whose DONE is newer
# than its FluidFile is skipped, so rerunning resumes the sweep.
#   ./runSweep.sh                      every case in runs/
#   ./runSweep.sh runs/M1.5_Re100_x0.14_y0.00 ...    only these
#   JOBS=2 NP=8 ./runSweep.sh          defaults (frg-n05 has 20 cores)
# To keep it running after closing the terminal:
#   nohup ./runSweep.sh > runSweep.log 2>&1 &
# Each case's AERO-F log is runs/<case>/log.out; postprocessing logs to log.postpro.
cd "$(dirname "$0")"
NP=${NP:-8}
JOBS=${JOBS:-2}
if [ $# -gt 0 ]; then cases=("$@"); else cases=(runs/*/FluidFile); cases=("${cases[@]%/FluidFile}"); fi

runCase() {
    local dir=$1 np=$2
    if [ "$dir"/DONE -nt "$dir"/FluidFile ]; then
        echo "$(date '+%F %T') skip  $dir (already done)"
        return
    fi
    echo "$(date '+%F %T') start $dir"
    if "$dir"/run.sh "$np" > /dev/null 2>&1 && "$dir"/postpro.sh > "$dir"/log.postpro 2>&1; then
        echo "$(date '+%F %T') done  $dir"
    else
        echo "$(date '+%F %T') FAIL  $dir (see $dir/log.out, $dir/log.postpro)"
    fi
}
export -f runCase
printf '%s\n' "${cases[@]}" | xargs -P "$JOBS" -I{} bash -c "runCase {} $NP"
