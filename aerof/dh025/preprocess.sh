#!/bin/bash
# Decompose dh025/sources/domain.top (written by ../dh025.py mesh) and run sower.
cd "$(dirname "$0")"
source ../tools.sh
mkdir -p data
$PARTNMESH sources/domain.top 96
$SOWER -fluid -mesh sources/domain.top -dec sources/domain.top.dec.96 \
    -cpu 16 -cpu 8 -cpu 1 -cluster 96 -output data/OUTPUT
