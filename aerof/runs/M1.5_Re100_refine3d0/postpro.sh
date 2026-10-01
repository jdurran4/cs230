#!/bin/bash
# Convert this case's AMR results to Exodus for ParaView.
#
# With AMR, AERO-F writes the refined mesh alongside every output frame NNNN:
#   results/<name>.con.NNNN, <name>.msh.NNNN.*, <name>.dec.NNNN.*, <QOI>.NNNN.*
# For each frame this rebuilds the mesh (postpro/<name>.top.NNNN), merges each QOI
# onto it (postpro/<QOI>.NNNN.xpost) and writes postpro/<name>.e-s.NNNN. Open the
# .e-s.* series in ParaView as one time series (the mesh can change between frames).
#
# Serial:    ./postpro.sh
# Parallel:  mpirun -np 12 bash postpro.sh   (frames split round-robin across ranks)
cd "$(dirname "$0")"
source ../../tools.sh

#module load netcdf/4.4.1.1

# Must match the names under Output/Postpro in the FluidFile
QOIs=(Mach Pressure Density Velocity Temperature)

RANK=${OMPI_COMM_WORLD_RANK:-0}
NTASKS=${OMPI_COMM_WORLD_SIZE:-1}

mkdir -p postpro

# Output frames, from the per-frame connectivity files (also handles restarts, where
# numbering does not start at 0001)
CON_FILES=(results/*.con.[0-9][0-9][0-9][0-9])
if [ ! -e "${CON_FILES[0]}" ]; then
    echo "No results/*.con.NNNN files found: nothing to postprocess"
    exit 1
fi
NAME=$(basename "${CON_FILES[0]}")
NAME=${NAME%.con.*}   # mesh name AERO-F used (from GeometryPrefix)
[ $RANK -eq 0 ] && echo "Mesh name $NAME, ${#CON_FILES[@]} frames, $NTASKS task(s)"

# Frame 0000: output written on the original (pre-AMR) mesh, if AERO-F wrote any
# (results/<QOI> without a frame number)
if [ $RANK -eq 0 ] && ls results | grep -E "^${QOIs[0]}" | grep -qvE "^${QOIs[0]}\.[0-9]{4}\."; then
    echo "Frame 0000 (original mesh) on rank $RANK"
    $SOWER -fluid -merge -con ../../data/OUTPUT.con -mesh ../../data/OUTPUT.msh \
        -dec ../../data/OUTPUT.dec -topo postpro/$NAME.top.0000 -quiet
    XPOSTS=()
    for QOI in "${QOIs[@]}"; do
        $SOWER -fluid -merge -con ../../data/OUTPUT.con -mesh ../../data/OUTPUT.msh \
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
