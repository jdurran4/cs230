#!/bin/bash
# Decompose the coarse fluid mesh (unsteady/mesh/domain.top from python -m sweep.meshGen)
# and run sower. Done once; every run in unsteady/runs/ reads the output through
# ../../data/OUTPUT.
cd "$(dirname "$0")"
source tools.sh
UNSTEADY=${CS230_DATA:-$HOME/Research/cs230-data}/unsteady
cd "$UNSTEADY"

# Specify number of subdomains, clusters
NSUB=96
NC=$NSUB

# Directory for sower outputs
DIR=data
mkdir -p "$DIR"

# Decompose fluid mesh
$PARTNMESH mesh/domain.top "$NSUB"

# Run Sower to pre-process fluid mesh. Each -cpu N writes a map for running on N MPI
# ranks (cluster: 12/24/48/96; workstation: 8 or 16)
$SOWER -fluid -mesh mesh/domain.top -dec mesh/domain.top.dec."$NSUB" \
    -cpu 96 -cpu 48 -cpu 24 -cpu 16 -cpu 12 -cpu 8 -cpu 1 -cluster "$NC" -output "$DIR"/OUTPUT
