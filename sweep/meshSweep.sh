#!/bin/bash
# Mesh-size study: oracles on the coarse meshes D/h in $MESHES, all with the p - p_inf
# source (sweep/oracle.py), then one dataset per mesh and kind. Includes 1.2: the original
# absolute-pressure runs in unsteady/oracle/ stay as they are and cannot be trained with
# these. Every mesh gets all 106 unsteady cases (the 8 of coarse_unsteady_8runs.npz + the
# 98 of batch2_names.txt) and all steady cases. M1.5_Re100_x0.04_y0.12 is one of the 8:
# leave it out of training with learn.train --exclude.
#
# Rerunning resumes: existing meshes and case directories are kept, and runOracles.sh
# skips cases that are DONE, so this also picks up after a one-case pilot.
#   sweep/meshSweep.sh                     everything, JOBS=12 NP=1
#   MESHES="2.4" JOBS=8 sweep/meshSweep.sh one mesh, 8 at a time
# Log: unsteady/oracle/meshSweep.log, plus runOracles.log in each oracle/Dh<X>/. To stop:
# kill -- -$(cat unsteady/oracle/meshSweep.pgid) (run it under setsid, or from systemd).
cd "$(dirname "$0")/.."
DATA=${CS230_DATA:-$HOME/Research/cs230-data}
ORACLE=$DATA/unsteady/oracle
export JOBS=${JOBS:-12} NP=${NP:-1} OMP_NUM_THREADS=1
MESHES=${MESHES:-0.6 1.2 2.4 4.8}
PY=.venv/bin/python
CASES="M1.3_Re36.8_x0.34_y0.22 M1.467_Re194.9_x0.27_y0.27 M1.5_Re100_x0.04_y0.12
       M1.665_Re20.3_x0.91_y0.03 M1.895_Re90.9_x0.73_y0.20 M2.016_Re56.2_x0.84_y0.03
       M2.187_Re124.1_x0.48_y0.12 M2.451_Re79.1_x0.76_y0.26
       $(cat "$ORACLE/batch2_names.txt")"   # the other 98 of the 106 unsteady cases

exec >> "$ORACLE/meshSweep.log" 2>&1
ps -o pgid= $$ | tr -d ' ' > "$ORACLE/meshSweep.pgid"
set -e
for Dh in $MESHES; do
    export MESH=Dh$Dh
    echo "$(date '+%F %T') === D/h $Dh"
    if [ ! -f "$ORACLE/$MESH/data/OUTPUT.1cpu" ]; then
        $PY -m sweep.oracle --Dh "$Dh" mesh
        bash sweep/oraclePreprocess.sh > /dev/null
    fi
    $PY -m sweep.oracle --Dh "$Dh" setup --new $CASES
    $PY -m sweep.oracle --Dh "$Dh" steady --new
    DIR=runs bash sweep/runOracles.sh
    DIR=steady bash sweep/runOracles.sh
    $PY -m learn.dataset unsteady --Dh "$Dh"
    $PY -m learn.dataset steady3d --Dh "$Dh"
done
echo "$(date '+%F %T') meshSweep finished"
