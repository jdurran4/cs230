#!/bin/bash
# Run and postprocess every coarse oracle case in unsteady/oracle/$DIR/ (DIR=runs, the
# default: unsteady oracles from python -m sweep.oracle setup; DIR=steady: steady oracles
# from python -m sweep.oracle steady), JOBS at a time with NP ranks each. A case whose DONE is
# newer than its FluidFile and oracleSource.txt is skipped, so rerunning resumes.
#   sweep/runOracles.sh                       every case
#   sweep/runOracles.sh M1.5_Re100_x0.04_y0.12 ...   only these
#   JOBS=4 NP=1 DIR=runs sweep/runOracles.sh  defaults
#   MESH=Dh2.4 sweep/runOracles.sh            mesh D/h = 2.4 (unsteady/oracle/Dh2.4/$DIR/)
# Log: unsteady/oracle/runOracles.log; each case's AERO-F log is $DIR/<case>/log.out.
cd "${CS230_DATA:-$HOME/Research/cs230-data}/unsteady/oracle/$MESH"
NP=${NP:-1}
JOBS=${JOBS:-4}
export DIR=${DIR:-runs}
if [ $# -gt 0 ]; then names=("$@"); else names=($(ls "$DIR")); fi

runCase() {
    local dir=$DIR/$1 np=$2
    if [ "$dir"/DONE -nt "$dir"/FluidFile ] && [ "$dir"/DONE -nt "$dir"/oracleSource.txt ]; then
        echo "$(date '+%F %T') skip  $dir (already done)"
        return
    fi
    echo "$(date '+%F %T') start $dir"
    if "$dir"/run.sh "$np" > /dev/null 2>&1; then
        bash "$dir"/postpro.sh > "$dir"/postpro.log 2>&1
        echo "$(date '+%F %T') done  $dir ($(grep -E '^It ' "$dir"/log.out | tail -1))"
    else
        echo "$(date '+%F %T') FAIL  $dir (see $dir/log.out)"
    fi
}
export -f runCase
printf '%s\n' "${names[@]}" | xargs -P "$JOBS" -I{} bash -c "runCase {} $NP" >> runOracles.log 2>&1
