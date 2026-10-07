"""One-off D/h = 0.25 oracle test (the cable regime): coarse mesh h = 4 D.

At D/h = 1.2 the oracle source reproduces the wake but not the flow deflection around
the body (blockage). This test repeats the oracle comparison where the body is much
smaller than a coarse cell. Everything lives in aerof/dh025/ (own sources/, data/,
runs/, oracle/), so the main pipeline (aerof/sources, runs/) is untouched; the shared
scripts are reused by overriding cylMeshGen's mesh parameters at import.

Mesh: h = 4 m, x in [-40, 80], y in [-40, 40] (D = 1), dz = 0.1 as before. Cylinder at
node (0, 0) + (0.15, 0.15) h (surface 0.35 m from the node), so the whole body sits in
one coarse column.

Steps (run from aerof/):
    python dh025.py mesh      write dh025/sources/domain.top, then: bash dh025/preprocess.sh
    python dh025.py case      write the resolved case dh025/runs/M1.5_Re100 (run it there)
    python dh025.py bin       after the resolved run + postpro: binTargets, binInputs,
                              and the oracle case dh025/oracle/M1.5_Re100
"""
import os
import re
import shutil
import sys

import cylMeshGen as mesh

# ---- mesh parameters (override the module's, shared by all binning code) ----------
mesh.h = 4.0
mesh.Lx, mesh.Ly = 120.0, 80.0
mesh.x0, mesh.y0 = 20.0, 0.0
here = os.path.dirname(os.path.abspath(__file__))
top = os.path.join(here, 'dh025')
mesh.outdir = os.path.join(top, 'sources')

case = 'M1.5_Re100'
ox, oy = 0.15, 0.15
xc, yc = 0.0 + ox * mesh.h, 0.0 + oy * mesh.h
refCase = os.path.join(here, 'runs', 'M1.5_Re100_x0.04_y0.12')   # same M, Re, D
caseDir = os.path.join(top, 'runs', case)
oracleDir = os.path.join(top, 'oracle', case)


def writeMesh():
    os.makedirs(mesh.outdir, exist_ok=True)
    nx, ny, fluid = mesh.fluidNodes()
    tets = mesh.fluidTets(nx, ny, fluid)
    faces = mesh.boundaryFaces(fluid, tets)
    mesh.writeDomain(os.path.join(mesh.outdir, 'domain.top'), fluid, tets, faces)
    print(f'dh025/sources/domain.top: {nx} x {ny} x 2 = {len(fluid)} nodes, {len(tets)} tets')
    with open(os.path.join(top, 'preprocess.sh'), 'w') as f:
        f.write(f"""#!/bin/bash
# Decompose dh025/sources/domain.top (written by ../dh025.py mesh) and run sower.
cd "$(dirname "$0")"
source ../tools.sh
mkdir -p data
$PARTNMESH sources/domain.top 96
$SOWER -fluid -mesh sources/domain.top -dec sources/domain.top.dec.96 \\
    -cpu 16 -cpu 8 -cpu 1 -cluster 96 -output data/OUTPUT
""")


def writeCase():
    dmin = mesh.writeCaseGeometry(caseDir, xc, yc, mesh.zOverhang)
    print(f'cylinder ({xc:g}, {yc:g}), min node distance {dmin:.3f} h')
    for f in ('run.sh', 'clean.sh', 'postpro.sh'):
        shutil.copy(os.path.join(refCase, f), caseDir)
    text = open(os.path.join(refCase, 'FluidFile')).read()
    # capsule axis start points: criterion [0] 1.5 D upstream, [2] and [3] 3 D upstream
    starts = iter([xc - 1.5, xc - 3.0, xc - 3.0])
    text = re.sub(r'Point_x = [^;]+;', lambda m: f'Point_x = {next(starts):g};', text)
    text = re.sub(r'Point_y = [^;]+;', f'Point_y = {yc:g};', text)
    # extra transition capsule (edge 1.0) so the size steps 4 -> 1 -> 0.2 -> 0.1
    extra = f"""
    // dh025: transition from the h = 4 coarse mesh to the 0.2 outer capsule [3]
    under SelectionCriterion[4] {{
        Type = APriori;
        PreRefine = On;
        MinimumEdgeLength = 1.0;
        NumLevels = 10;
        under RefinementDomain {{
            Type = Capsule;
            under Capsule {{
                Point_x = {xc - 6:g};
                Point_y = {yc:g};
                Point_z = 0.05;
                Normal_x = 1;
                Normal_y = 0;
                Normal_z = 0;
                Radius = 8.0;
                Length = 25.0;
            }}
        }}
    }}

    InitializeRefinementTag = On;"""
    text = text.replace('\n    InitializeRefinementTag = On;', extra, 1)
    # The h = 4 x 4 x 0.1 coarse tets are very flat: with the default aspect-ratio limit
    # (1000) arbitrary bisection refused almost every split (9 edges refined). Raise it,
    # and allow more levels (4 -> 0.02 m takes ~2 bisections per halving in-plane).
    text = text.replace('// MaximumAspectRatio = 1000;', 'MaximumAspectRatio = 1.0e6;   // dh025')
    text = re.sub(r'NumLevels = \d+;', 'NumLevels = 25;', text)
    header = (f'// dh025 test (D/h = 0.25, coarse h = 4): resolved case {case}, written by '
              f'aerof/dh025.py\n// from {os.path.relpath(refCase, here)}/FluidFile with the '
              f'cylinder at ({xc:g}, {yc:g}) and an extra\n// transition capsule [4].\n')
    open(os.path.join(caseDir, 'FluidFile'), 'w').write(header + text)
    print('wrote', caseDir)


def binAndOracle():
    import numpy as np
    import binOneCell as bin1
    import binInputs
    import binTargets
    binTargets.runsDir = binInputs.runsDir = os.path.join(top, 'runs')
    nodes, tets = bin1.readDomain(os.path.join(mesh.outdir, 'domain.top'))
    binInputs.binInputs(case, nodes, tets)          # binTargets reads inputs.npz
    binTargets.binTargets(case, nodes, tets)

    # oracle case: the main oracle FluidFile works unchanged (same freestream; the mesh
    # path ../../data/OUTPUT now resolves to dh025/data)
    os.makedirs(oracleDir, exist_ok=True)
    mainOracle = os.path.join(here, 'oracle', 'M1.5_Re100_x0.04_y0.12')
    for f in ('FluidFile', 'run.sh', 'clean.sh', 'postpro.sh'):
        shutil.copy(os.path.join(mainOracle, f), oracleDir)
    t = np.load(os.path.join(caseDir, 'targets.npz'))
    S = t['S'][-1]                                       # force on the fluid per span [N/m]
    f = S / mesh.h ** 2                                  # force density [N/m^3]
    np.savetxt(os.path.join(oracleDir, 'oracleSource.txt'),
               np.column_stack([t['columns'], f]), fmt='%.16e')
    q = float(t['q'])
    print(f'oracle: {len(S)} columns, Cd = {-S[:, 0].sum() / (q * mesh.D):.4f}, '
          f'Cl = {-S[:, 1].sum() / (q * mesh.D):.4f} -> {oracleDir}')


if __name__ == '__main__':
    {'mesh': writeMesh, 'case': writeCase, 'bin': binAndOracle}[sys.argv[1]]()
