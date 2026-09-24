#!/bin/bash

PARTNMESH_EXECUTABLE=/home/rtezaur/bin/partnmesh
SOWER_EXECUTABLE=/home/rtezaur/bin/sower

# Specify number of subdomains, processes, clusters
NSUB=96
NC=$NSUB

# Directory for sower outputs
DIR=data

# Decompose fluid mesh
$PARTNMESH_EXECUTABLE sources/domain_ref.top "$NSUB"

# Run Sower to pre-process fluid mesh
$SOWER_EXECUTABLE -fluid -mesh sources/domain_ref.top -dec sources/domain_ref.top.dec."$NSUB" \
    -cpu 96 -cpu 48 -cpu 24 -cpu 12 -cpu 1 -cluster "$NC" -output "$DIR"/OUTPUT