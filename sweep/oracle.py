"""
Unsteady coarse oracle runs: replay each resolved run's true source S(t) on the coarse
mesh, from the same impulsive start, so the coarse states the network trains on are the
ones it will see online (with the exact source instead of the network).

Coarse mesh: the sweep's x-y grid (meshGen, h = D/1.2) extruded to nLayers layers of
thickness hz, each hex split into the same 6 tets as meshGen. No cylinder, no AMR.

Source: the resolved flow is z-uniform, so a surface station at (x, y) carries a force
per unit span s(t) (on the fluid, absolute pressure + shear, as learn/pilot3d.py), and
the force on the fluid in the 3D median-dual cell K of coarse node K is
    F_K(t) = sum_s s(t) L_K(x_s, y_s),
with L_K the length of the vertical line through the station inside cell K. The interior
cells (levels 1..nLayers-1) of a column are identical; the boundary half cells (levels 0
and nLayers) split the column's force unevenly, which made the coarse flow z-dependent.
So every level of a column gets the interior cells' force per unit volume f = F_K / V_K
(z-uniform, column total unchanged), which is also how the online neural source applies
its prediction. AERO-F adds -ctrlVol f to the momentum residual (ctrlVol = V_K: no
embedded surface in the coarse run).

oracleSource.txt (SI units, read by AERO-F's OracleSourceFile):
    nPoints nTimes
    x y z                    nPoints lines: the coarse nodes that get a source (every
                             level of each column the body touches)
    t                        then nTimes blocks: the time [s] of a resolved snapshot,
    fx fy                    followed by nPoints lines of f_K [N/m^3] in the same order
AERO-F interpolates f linearly in time between the snapshots (held constant before the
first and after the last). The same data, plus F_K and the node ids, is saved in
oracleSource.npz for building the training targets.

Usage (from the repo root):
    python -m sweep.oracle mesh               unsteady/oracle/mesh/domain.top
    sweep/oraclePreprocess.sh                 decompose it (unsteady/oracle/data/)
    python -m sweep.oracle setup [case ...]   oracle/runs/<case>/ for every resolved run
                                              with DONE (or only the listed ones)
    sweep/runOracles.sh                       run + postpro them
    python -m sweep.oracle steady [case ...]  oracle/steady/<case>/ for every steady-v0
                                              resolved run (or only the listed ones)
    DIR=steady sweep/runOracles.sh            run + postpro them

Steady oracles: the same coarse mesh and z-uniform binning, with the constant source of
a converged steady resolved run (steady-v0/runs/<case>, last output; written as two
identical snapshots), solved to steady state (SteadyOracleFluidFile_template). They add
case coverage (~300 (M, Re, position) cases) for the settled flow.
"""
import os
import re
import shutil
import sys

import numpy as np

import common
from learn import pilot3d
from sweep import meshGen, newRun

# ============= User-defined parameters ==============================================
nLayers = 4                # coarse layers in z
hz = meshGen.h             # layer thickness (any value: the flow is z-uniform)
nzSample = 64              # z samples per layer for L_K
outputEvery = 0.25         # field snapshots, D/U (the resolved runs': 5 steps)
restartEvery = 10.0        # restart files, D/U

oracleDir = os.path.join(common.UNSTEADY, 'oracle')
meshFile = os.path.join(oracleDir, 'mesh', 'domain.top')
sweepDir = os.path.dirname(os.path.abspath(__file__))
templateFile = os.path.join(sweepDir, 'OracleFluidFile_template')
steadyDir = os.path.join(oracleDir, 'steady')
steadyTemplateFile = os.path.join(sweepDir, 'SteadyOracleFluidFile_template')
caseScripts = ('run.sh', 'clean.sh', 'postpro.sh')


# ============= Coarse 3D mesh =======================================================
def writeMesh():
    nx, ny, nodes = meshGen.fluidNodes(nLayers, hz)
    tets = meshGen.fluidTets(nx, ny, nodes, nLayers)
    faces = meshGen.boundaryFaces(nodes, tets, nLayers * hz)
    os.makedirs(os.path.dirname(meshFile), exist_ok=True)
    meshGen.writeDomain(meshFile, nodes, tets, faces)
    print(f'{meshFile}: {nx} x {ny} x {nLayers + 1} = {len(nodes)} nodes, {len(tets)} tets, '
          f'h = {meshGen.h:.4f}, hz = {hz:.4f}')


# ============= Binning ==============================================================
def stationSourceHistory(runDir):
    """Station x-y (ns, 2), snapshot times (nt,) and the source on the fluid per unit
    span at every snapshot, absolute pressure (nt, ns, 2) [N/m] (pilot3d.stationSources
    for every snapshot instead of the last)."""
    geo = meshGen.loadGeometry(runDir)
    X = pilot3d.readSurfaceNodes(geo['surfFile'])
    t, Fs = pilot3d.readSurfaceForce(os.path.join(runDir, 'postpro', 'EmbeddedSurfaceForce'))
    xy, station = np.unique(np.round(X[:, :2], 12), axis=0, return_inverse=True)
    Fst = np.zeros((len(t), len(xy), 2))
    np.add.at(Fst, (slice(None), station.ravel()), Fs[:, :, :2])
    _, _, pInf = common.readInlet(os.path.join(runDir, 'FluidFile'))
    s = -(Fst - pInf * geo['span'] * pilot3d.stationNormals(xy)) / geo['span']
    if np.any(np.diff(t) <= 0):
        raise ValueError(f'{runDir}: snapshot times are not increasing')
    return xy, t, s


def binCase(name, nodes, tets, V, steady=False):
    """Write oracle/runs/<name>/oracleSource.{txt,npz} (steady: the steady-v0 run's last
    output, held constant, to oracle/steady/<name>/)."""
    runDir = os.path.join(common.STEADY if steady else common.UNSTEADY, 'runs', name)
    xy, t, s = stationSourceHistory(runDir)
    if steady:                                             # AERO-F needs >= 2 snapshots
        t, s = np.array([0.0, 1.0]), s[[-1, -1]]
    thick = nLayers * hz
    L = pilot3d.lengthsInCells(xy, nodes, tets, thick, nLayers * nzSample)   # (ns, nn)
    hit = np.flatnonzero(L.sum(axis=0) > 0)                # nodes whose cell holds a station
    F3 = np.einsum('sk,tsc->tkc', L[:, hit], s)            # (nt, nhit, 2) true 3D cell force [N]

    # z-uniform source: every level of a column gets the interior cells' force per unit
    # volume (the interior cells are identical; the boundary half cells' own split is
    # uneven), the same way the online neural source applies it. Column totals unchanged.
    level = np.rint(nodes[:, 2] / hz).astype(int)
    cols = np.unique(np.round(nodes[hit, :2], 9), axis=0)
    colOf = np.full(len(nodes), -1)
    for c, (cx, cy) in enumerate(cols):
        colOf[np.hypot(nodes[:, 0] - cx, nodes[:, 1] - cy) < 1e-6] = c
    k = np.flatnonzero(colOf >= 0)
    k = k[np.lexsort((level[k], colOf[k]))]                # by column, then level
    inner = np.isin(hit, k[(level[k] > 0) & (level[k] < nLayers)])
    fInner = np.zeros((len(t), len(cols), 2))
    nInner = np.zeros(len(cols))
    for j in np.flatnonzero(inner):
        fInner[:, colOf[hit[j]]] += F3[:, j] / V[hit[j]]
        nInner[colOf[hit[j]]] += 1
    fInner /= np.maximum(nInner, 1)[None, :, None]
    spread = max((np.abs(F3[:, j] / V[hit[j]] - fInner[:, colOf[hit[j]]]).max()
                  for j in np.flatnonzero(inner)), default=0) / np.abs(fInner).max()
    f = fInner[:, colOf[k]]                                # (nt, nk, 2) [N/m^3]
    F = f * V[k, None]                                     # (nt, nk, 2) force [N]

    err = np.abs(F.sum(axis=1) - thick * s.sum(axis=1)).max() / np.abs(thick * s.sum(axis=1)).max()
    d = os.path.join(steadyDir if steady else os.path.join(oracleDir, 'runs'), name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'oracleSource.txt'), 'w') as fh:
        fh.write(f'{len(k)} {len(t)}\n')
        np.savetxt(fh, nodes[k], fmt='%.16e')
        for n in range(len(t)):
            fh.write(f'{t[n]:.16e}\n')
            np.savetxt(fh, f[n], fmt='%.16e')
    np.savez(os.path.join(d, 'oracleSource.npz'), node=k, xyz=nodes[k], level=level[k],
             col=colOf[k], t=t, F=F, f=f, V=V[k], hz=hz, nLayers=nLayers)
    print(f'{os.path.relpath(d, oracleDir)}: {len(cols)} columns x {nLayers + 1} levels = {len(k)} source '
          f'nodes, {len(t)} snapshots (t = {t[0]:.3g} .. {t[-1]:.4g} s), conservation error '
          f'{err:.1e}, interior-level spread {spread:.1e}')


# ============= Case directories =====================================================
def parseCase(name):
    m = re.fullmatch(r'M([\d.]+)_Re([\d.]+)_x([\d.]+)_y([\d.]+)', name)
    return newRun.Case(*(float(g) for g in m.groups()))


def substitutions(caseDir, c):
    fs = newRun.freestream(c.mach, c.Re)
    inlet = ('BoundaryConditions', 'Inlet')
    return {
        (('Input',), 'GeometryPrefix'): newRun.quoted(os.path.join(oracleDir, 'data', 'OUTPUT'), caseDir),
        (inlet, 'Mach'): f'{c.mach:.10g}',
        (inlet, 'Density'): f'{fs["rho"]:.10e}',
        (inlet, 'Pressure'): f'{fs["p"]:.10e}',
        (('Time',), 'TimeStep'): f'{fs["tRef"] / newRun.stepsPerDU:.10e}',
        (('Time',), 'MaxTime'): f'{fs["tRef"] * newRun.tEnd:.10e}',
        (('Time',), 'MaxIts'): f'{newRun.steps(newRun.tEnd) + 100}',
        (('Output', 'Postpro'), 'Frequency'): f'{newRun.steps(outputEvery)}',
        (('Output', 'Restart'), 'Frequency'): f'{newRun.steps(restartEvery)}',
    }


def caseInfo(name, c):
    fs = newRun.freestream(c.mach, c.Re)
    return f"""\
// Oracle for runs/{name}: Mach = {c.mach:g}, Re_D = {c.Re:g}   (written by oracle.py)
//   coarse mesh: {nLayers} layers of hz = {hz:.6g} m; source from oracleSource.txt
//   U = {fs['U']:.6g} m/s, rho = {fs['rho']:.6g} kg/m^3, p = {fs['p']:.6g} Pa
//   D/U = {fs['tRef']:.6g} s; dt = D/U / {newRun.stepsPerDU}, t_end = {newRun.tEnd:g} D/U, snapshots every {outputEvery:g} D/U"""


def setupCase(name, nodes, tets, V):
    c = parseCase(name)
    d = os.path.join(oracleDir, 'runs', name)
    os.makedirs(d, exist_ok=True)
    binCase(name, nodes, tets, V)
    for sub in ('results', 'references', 'postpro'):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    for f in caseScripts:
        shutil.copy2(os.path.join(sweepDir, 'runTemplate', f), d)
    with open(templateFile) as fh:
        text = newRun.fillTemplate(fh.read(), substitutions(d, c), caseInfo(name, c))
    with open(os.path.join(d, 'FluidFile'), 'w') as fh:
        fh.write(text)


def steadySetupCase(name, nodes, tets, V):
    """oracle/steady/<name>/ from the steady-v0 resolved run, with its own inlet values
    (steady-v0 used slightly different freestream constants than newRun.freestream)."""
    runDir = os.path.join(common.STEADY, 'runs', name)
    mach, rho, p = common.readInlet(os.path.join(runDir, 'FluidFile'))
    d = os.path.join(steadyDir, name)
    os.makedirs(d, exist_ok=True)
    binCase(name, nodes, tets, V, steady=True)
    for sub in ('results', 'references', 'postpro'):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    for f in caseScripts:
        shutil.copy2(os.path.join(sweepDir, 'runTemplate', f), d)
    inlet = ('BoundaryConditions', 'Inlet')
    subs = {(('Input',), 'GeometryPrefix'): newRun.quoted(os.path.join(oracleDir, 'data', 'OUTPUT'), d),
            (inlet, 'Mach'): f'{mach:.10g}',
            (inlet, 'Density'): f'{rho:.10e}',
            (inlet, 'Pressure'): f'{p:.10e}'}
    info = (f'// Steady oracle for steady-v0/runs/{name}: Mach = {mach:g}   (written by oracle.py)\n'
            f'//   coarse mesh: {nLayers} layers of hz = {hz:.6g} m; constant source from oracleSource.txt\n'
            f'//   rho = {rho:.6g} kg/m^3, p = {p:.6g} Pa (the resolved run\'s inlet)')
    with open(steadyTemplateFile) as fh:
        text = newRun.fillTemplate(fh.read(), subs, info)
    with open(os.path.join(d, 'FluidFile'), 'w') as fh:
        fh.write(text)


def steadyNames():
    """steady-v0 resolved runs that finished, with a case name parseCase understands
    (this drops the _me2M / zo0.1 duplicate runs)."""
    runs = os.path.join(common.STEADY, 'runs')
    ok = lambda n: (os.path.exists(os.path.join(runs, n, 'DONE'))
                    and re.fullmatch(r'M[\d.]+_Re[\d.]+_x[\d.]+_y[\d.]+', n))
    return sorted(n for n in os.listdir(runs) if ok(n))


# ============= Main =================================================================
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('mesh', 'setup', 'steady'):
        sys.exit(__doc__)
    if sys.argv[1] == 'mesh':
        writeMesh()
        sys.exit()
    nodes, tets = common.readTop(meshFile)
    V = pilot3d.dualVolumes(nodes, tets)
    if sys.argv[1] == 'steady':
        for name in sys.argv[2:] or steadyNames():
            steadySetupCase(name, nodes, tets, V)
        sys.exit()
    names = sys.argv[2:] or sorted(
        n for n in os.listdir(os.path.join(common.UNSTEADY, 'runs'))
        if os.path.exists(os.path.join(common.UNSTEADY, 'runs', n, 'DONE')))
    for name in names:
        setupCase(name, nodes, tets, V)
