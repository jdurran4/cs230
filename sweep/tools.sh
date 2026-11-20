# AERO tool locations, sourced by preprocess.sh and each run's run.sh / postpro.sh.
# Picks the tools for the machine it runs on, so the same file can be synced to the cluster.
# Paths below are specific to the Stanford FRG machines: edit them for your own AERO-F build.
case "$(hostname)" in
    independence2*|ind2-*)
        # Independence2: rtezaur's builds (Intel 2018; run.sbatch loads the modules).
        # This AERO-F predates the Refine3d -> Refine3D keyword rename (Nov 2023).
        AEROF=/home/rtezaur/bin/aerof.opt
        PARTNMESH=/home/rtezaur/bin/partnmesh
        SOWER=/home/rtezaur/bin/sower
        XP2EXO=/home/pavery/bin/xp2exo
        ;;
    *)
        # Workstation: own builds from ~/frg (see ~/frg/frg_install_workstation.txt)
        FRG_HOME=${FRG_HOME:-$HOME/frg}
        AEROF=$FRG_HOME/aero-f/build/bin/aerof.opt
        PARTNMESH=$FRG_HOME/metis/build/bin/partnmesh
        SOWER=$FRG_HOME/sower/build/bin/sower
        XP2EXO=$FRG_HOME/xp2exo/build/bin/xp2exo
        ;;
esac
