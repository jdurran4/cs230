AEROF=/home/tdurrant/aero-f/build/bin/aerof.opt
NP=48 # number of mpi processes

./clean.sh

# run simulation
# module load cmake/3.8.1 gcc/9.1.0 openmpi/4.1.2 imkl/2019
mpirun -np $NP $AEROF FluidFile |& tee log.out