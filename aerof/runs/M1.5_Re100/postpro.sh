#!/bin/bash

XP2EXO=/home/pavery/bin/xp2exo
SOWER=/home/pavery/bin/sower

#module load netcdf/4.4.1.1

$SOWER -fluid -merge -con ../../data/OUTPUT.con -mesh ../../data/OUTPUT.msh \
        -result results/Mach.bin  -output postpro/Mach -name Mach
$SOWER -fluid -merge -con ../../data/OUTPUT.con -mesh ../../data/OUTPUT.msh \
        -result results/Pressure.bin -output postpro/Pressure  -name Pressure
$SOWER -fluid -merge -con ../../data/OUTPUT.con -mesh ../../data/OUTPUT.msh \
        -result results/Velocity.bin -output postpro/Velocity  -name Velocity
$SOWER -fluid -merge -con ../../data/OUTPUT.con -mesh ../../data/OUTPUT.msh \
        -result results/Density.bin -output postpro/Density  -name Density
$SOWER -fluid -merge -con ../../data/OUTPUT.con -mesh ../../data/OUTPUT.msh \
        -result results/Temperature.bin -output postpro/Temperature  -name Temperature
$SOWER -fluid -merge -con ../../data/OUTPUT.con -mesh ../../data/OUTPUT.msh \
        -result results/Pressure.bin -skin postpro/surfacePressure -bc -3

$XP2EXO ../../sources/domain_ref.top postpro/domain_ref.exo postpro/Pressure.xpost postpro/Mach.xpost postpro/Velocity.xpost postpro/Density.xpost postpro/Temperature.xpost