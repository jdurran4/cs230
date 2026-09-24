#!/bin/bash
# Convert this case's results/*.bin to postpro/domain.exo for ParaView.
#
# NOTE: the merge below uses the coarse sower mesh (data/OUTPUT, sources/domain.top).
# With AMR pre-refinement the solution lives on the refined mesh, so this needs to be
# pointed at the refined mesh AERO-F writes before it will line up.
cd "$(dirname "$0")"
source ../../tools.sh

#module load netcdf/4.4.1.1

mkdir -p postpro
for f in Mach Pressure Velocity Density Temperature; do
    $SOWER -fluid -merge -con ../../data/OUTPUT.con -mesh ../../data/OUTPUT.msh \
        -result results/$f.bin -output postpro/$f -name $f
done

# Cylinder surface quantities: the cylinder is an embedded surface, not a fluid mesh
# boundary, so they come from AERO-F's embedded-surface outputs, not sower -skin.

$XP2EXO ../../sources/domain.top postpro/domain.exo postpro/Pressure.xpost postpro/Mach.xpost \
    postpro/Velocity.xpost postpro/Density.xpost postpro/Temperature.xpost
