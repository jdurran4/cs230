#!/bin/bash
# Convert this case's AMR results to Exodus for ParaView.
#
# With AMR, AERO-F writes the refined mesh alongside every output frame NNNN:
#   results/<name>.con.NNNN, <name>.msh.NNNN.*, <name>.dec.NNNN.*, <QOI>.NNNN.*
# For each frame this rebuilds the mesh (postpro/<name>.top.NNNN), merges each QOI
# onto it (postpro/<QOI>.NNNN.xpost) and writes postpro/<name>.e-s.NNNN. Open the
# .e-s.* series in ParaView as one time series (the mesh can change between frames).
# Output written before any refinement in this run has no frame number; it is on the
# mesh the FluidFile reads (GeometryPrefix, the coarse mesh) and becomes frame 0000.
#
# The embedded surface (Input/EmbeddedSurface) goes to postpro/embeddedSurface.exo with
# the per-node force postpro/EmbeddedSurfaceForce (force ON the body, p - p_inf + shear,
# integrated over the whole tube) as the nodal field "Force", the same way the lab's
# FSI_NAS PostProcess.FSI.structure.sh attaches EmbeddedSurfaceDisplacement.
#
# Serial:    ./postpro.sh
# Parallel:  mpirun -np 12 bash postpro.sh   (frames split round-robin across ranks)
cd "$(dirname "$0")"
ROOT=$PWD
until [ -f "$ROOT/tools.sh" ] || [ "$ROOT" = / ]; do ROOT=$(dirname "$ROOT"); done
source "$ROOT/tools.sh"
GEOM=$(sed -n 's/^[[:space:]]*GeometryPrefix[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' FluidFile)
SURF=$(sed -n 's/^[[:space:]]*EmbeddedSurface[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' FluidFile)

#module load netcdf/4.4.1.1

# Must match the names under Output/Postpro in the FluidFile
QOIs=(Mach Pressure Density Velocity Temperature)

RANK=${OMPI_COMM_WORLD_RANK:-0}
NTASKS=${OMPI_COMM_WORLD_SIZE:-1}

mkdir -p postpro

# Output frames, from the per-frame connectivity files (also handles restarts, where
# numbering does not start at 0001)
CON_FILES=(results/*.con.[0-9][0-9][0-9][0-9])
[ -e "${CON_FILES[0]}" ] || CON_FILES=()
NAME=$(basename "$GEOM")   # mesh name AERO-F uses (from GeometryPrefix)
HAS0000=0
ls results | grep -E "^${QOIs[0]}" | grep -qvE "^${QOIs[0]}\.[0-9]{4}\." && HAS0000=1
if [ ${#CON_FILES[@]} -eq 0 ] && [ $HAS0000 -eq 0 ]; then
    echo "No output in results/: nothing to postprocess"
    exit 1
fi
[ $RANK -eq 0 ] && echo "Mesh name $NAME, frame 0000: $HAS0000, ${#CON_FILES[@]} numbered frames, $NTASKS task(s)"

# Frame 0000: output on the mesh the run started from (results/<QOI> without a frame number)
if [ $RANK -eq 0 ] && [ $HAS0000 -eq 1 ]; then
    echo "Frame 0000 (starting mesh $GEOM) on rank $RANK"
    $SOWER -fluid -merge -con $GEOM.con -mesh $GEOM.msh \
        -dec $GEOM.dec -topo postpro/$NAME.top.0000 -quiet
    XPOSTS=()
    for QOI in "${QOIs[@]}"; do
        $SOWER -fluid -merge -con $GEOM.con -mesh $GEOM.msh \
            -result results/$QOI -output postpro/$QOI.0000 -name $QOI -quiet
        XPOSTS+=(postpro/$QOI.0000.xpost)
    done
    $XP2EXO postpro/$NAME.top.0000 postpro/$NAME.e-s.0000 "${XPOSTS[@]}"
fi

for k in "${!CON_FILES[@]}"; do
    [ $((k % NTASKS)) -eq $RANK ] || continue
    i=${CON_FILES[$k]##*.}
    echo "Frame $i on rank $RANK"

    $SOWER -fluid -merge -con results/$NAME.con.$i -mesh results/$NAME.msh.$i. \
        -dec results/$NAME.dec.$i. -topo postpro/$NAME.top.$i -quiet

    XPOSTS=()
    for QOI in "${QOIs[@]}"; do
        $SOWER -fluid -merge -con results/$NAME.con.$i -mesh results/$NAME.msh.$i. \
            -result results/$QOI.$i. -output postpro/$QOI.$i -name $QOI -quiet
        XPOSTS+=(postpro/$QOI.$i.xpost)
    done

    $XP2EXO postpro/$NAME.top.$i postpro/$NAME.e-s.$i "${XPOSTS[@]}"
done

# Embedded surface with the per-node force
if [ $RANK -eq 0 ] && [ -f postpro/EmbeddedSurfaceForce ]; then
    echo "Embedded surface $SURF on rank $RANK"
    $XP2EXO $SURF postpro/embeddedSurface.exo postpro/EmbeddedSurfaceForce
fi
