#!/bin/bash
# Decompose the coarse oracle mesh (unsteady/oracle/mesh/domain.top from
# python -m sweep.oracle mesh) and run sower. Done once; every case in unsteady/oracle/runs/
# reads the output through ../../data/OUTPUT. The mesh is small (~7k nodes), so few
# subdomains: run each case on 1, 2, 4 or 8 ranks.
cd "$(dirname "$0")"
source tools.sh
# MESH=Dh2.4 selects the mesh of python -m sweep.oracle --Dh 2.4 mesh.
cd "${CS230_DATA:-$HOME/Research/cs230-data}/unsteady/oracle/$MESH"
NSUB=8
mkdir -p data
$PARTNMESH mesh/domain.top "$NSUB"
$SOWER -fluid -mesh mesh/domain.top -dec mesh/domain.top.dec."$NSUB" \
    -cpu 8 -cpu 4 -cpu 2 -cpu 1 -cluster "$NSUB" -output data/OUTPUT
