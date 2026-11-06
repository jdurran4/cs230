#!/bin/bash
# Turn ParaView animation frames (PREFIX.0000.png, PREFIX.0001.png, ...) into an mp4.
# One movie per frame prefix found in the input folder, named PREFIX.mp4.
#
# Usage:  online/pngsToMp4.sh [IN_DIR] [OUT_DIR] [FPS]
#   IN_DIR   default ~/Research/cs230-data/paraview_pngs
#   OUT_DIR  default ~/Research/cs230-data/unsteady/online/slides (next to the slide figures)
#   FPS      default 30
# H.264 + yuv420p plays in PowerPoint, Keynote, Google Slides and browsers; the pad filter
# rounds odd frame sizes up to even, which yuv420p requires.
IN=${1:-~/Research/cs230-data/paraview_pngs}
OUT=${2:-~/Research/cs230-data/unsteady/online/slides}
FPS=${3:-30}
mkdir -p "$OUT"
for prefix in $(ls "$IN" | sed -n 's/\.[0-9]\{4\}\.png$//p' | sort -u); do
  n=$(ls "$IN"/"$prefix".[0-9][0-9][0-9][0-9].png | wc -l)
  echo "$prefix: $n frames at $FPS fps -> $OUT/$prefix.mp4"
  ffmpeg -loglevel error -y -framerate "$FPS" -i "$IN/$prefix.%04d.png" \
         -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -crf 18 \
         -movflags +faststart "$OUT/$prefix.mp4"
done
