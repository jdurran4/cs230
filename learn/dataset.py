"""Coarse-state dataset: one row per source column of every coarse oracle run.

Each resolved steady run's exact binned source (grid 0, the simulation grid, D/h = 1.2)
was replayed on the coarse mesh (the coarse oracle runs, steady-v0/coarse/<run>). The
network online sees the coarse solver's node state, so the rows pair
  the coarse oracle's converged state at the column's nodes     (what the network sees)
  the true source S of the column, from the resolved run         (what it must predict)

Everything is stored raw, in SI units and in the solver's x-y frame; learn/features.py
turns it into network inputs and targets, so the frame and normalization are choices
made there, not baked in here.

Per row:
  run (index into runNames), machInf, ReInf, ox, oy, qInf, duplicate   case
  x, y, h, D                          column grid point, coarse spacing, diameter
  volFrac, Nx, Ny, rx, ry             column geometry: solid fraction of the column,
                                      outward area normal of the surface in it / D,
                                      body-piece centroid minus grid point [m]
  rho, u, v, p                        node state (mean of the column's 2 z-levels)
  drho_dx, drho_dy, du_dx, ...        nodal gradients of rho, u, v, p (same mean), as
                                      AERO-F computes them: unweighted least squares
                                      over the edge neighbors (Gradient = LeastSquares)
  Sx, Sy                              true force on the fluid per unit span [N/m]

Usage (from the repo root):
  python -m learn.dataset            -> DATA/datasets/coarse_steady.npz
  python -m learn.dataset unsteady   -> DATA/datasets/coarse_unsteady.npz (buildOracle:
                                        unsteady oracle snapshots, interior 3D cells)
  python -m learn.dataset steady3d   -> DATA/datasets/coarse_steady3d.npz (buildOracle:
                                        steady oracles on the same 4-layer mesh)
"""
import os

import numpy as np

import common

coarseDir = os.path.join(common.STEADY, 'coarse')
runsDir = os.path.join(common.STEADY, 'runs')
outFile = os.path.join(common.DATA, 'datasets', 'coarse_steady.npz')
stateNames = ['rho', 'u', 'v', 'p']


def leastSquaresGradient(nodes, tets):
    """Function V (nn, k) -> gradients (nn, k, 3), the way AERO-F computes them with
    Gradient = LeastSquares: per node i, minimize sum over edge neighbors j of
    (V_j - V_i - grad V_i . dx_ij)^2, i.e.  grad V_i = A_i^-1 sum_j dx_ij (V_j - V_i)
    with A_i = sum_j dx_ij dx_ij^T."""
    pairs = np.concatenate([tets[:, [a, b]] for a in range(4) for b in range(a + 1, 4)])
    edges = np.unique(np.sort(pairs, axis=1), axis=0)
    i = np.concatenate([edges[:, 0], edges[:, 1]])        # each edge serves both ends
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    dx = nodes[j] - nodes[i]                              # (2 ne, 3)
    A = np.zeros((len(nodes), 3, 3))
    np.add.at(A, i, dx[:, :, None] * dx[:, None, :])
    Ainv = np.linalg.inv(A)

    def gradient(V):
        b = np.zeros((len(nodes), V.shape[1], 3))
        np.add.at(b, i, (V[j] - V[i])[:, :, None] * dx[:, None, :])
        return np.einsum('nab,nkb->nka', Ainv, b)
    return gradient


def converged(runDir):
    logFile = os.path.join(runDir, 'log.out')
    if not os.path.isfile(logFile) or not os.path.isdir(os.path.join(runDir, 'postpro')):
        return False
    res, aborted = common.residualHistory(logFile)
    return len(res) > 0 and res[-1] <= 1e-5 and not aborted


def build():
    names = sorted(n for n in os.listdir(coarseDir)
                   if os.path.isfile(os.path.join(runsDir, n, 'samples.npz')))
    ok = [n for n in names if converged(os.path.join(coarseDir, n))]

    # the coarse mesh is the same for every run: columns and gradient operator once
    post = lambda n: os.path.join(coarseDir, n, 'postpro')
    nodes, tets = common.readTop(os.path.join(post(ok[0]), 'OUTPUT.top.0000'))
    colXY, colOf = np.unique(np.round(nodes[:, :2], 9), axis=0, return_inverse=True)
    colOf = colOf.ravel()
    perCol = np.bincount(colOf)
    colMean = lambda a: np.stack([np.bincount(colOf, a[:, k]) for k in range(a.shape[1])],
                                 axis=1) / perCol[:, None]
    gradient = leastSquaresGradient(nodes, tets)

    rows = {}
    add = lambda k, v: rows.setdefault(k, []).append(v)
    for k, name in enumerate(ok):
        read = lambda q: common.readXpost(os.path.join(post(name), f'{q}.0000.xpost'))[1][-1]
        vel = read('Velocity')
        V = np.column_stack([read('Density'), vel[:, 0], vel[:, 1], read('Pressure')])
        G = gradient(V)                                               # (nn, 4, 3)
        Vc = colMean(V)                                               # (ncol, 4)
        Gc = colMean(G[:, :, :2].reshape(len(V), 8)).reshape(-1, 4, 2)

        s = np.load(os.path.join(runsDir, name, 'samples.npz'))
        for r in np.flatnonzero(s['grid'] == 0):
            c = np.argmin(np.hypot(colXY[:, 0] - s['colX'][r], colXY[:, 1] - s['colY'][r]))
            assert np.hypot(*(colXY[c] - [s['colX'][r], s['colY'][r]])) < 1e-3  # .top: 5 digits
            # samples.npz stores N and r in the flow frame of the FINE solution: back to x-y
            e = np.array([s['ePar_x'][r], s['ePar_y'][r]])
            eP = np.array([-e[1], e[0]])
            N = s['Npar'][r] * e + s['Nperp'][r] * eP
            rv = (s['rpar'][r] * e + s['rperp'][r] * eP) * s['h'][r]
            for key in ('machInf', 'ReInf', 'ox', 'oy', 'qInf', 'duplicate', 'h', 'D',
                        'volFrac', 'Sx', 'Sy'):
                add(key, s[key][r])
            add('run', k)
            add('x', s['colX'][r])
            add('y', s['colY'][r])
            add('Nx', N[0])
            add('Ny', N[1])
            add('rx', rv[0])
            add('ry', rv[1])
            for m, q in enumerate(stateNames):
                add(q, Vc[c, m])
                add(f'd{q}_dx', Gc[c, m, 0])
                add(f'd{q}_dy', Gc[c, m, 1])

    out = {key: np.array(v, dtype=float) for key, v in rows.items()}
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    np.savez(outFile, runNames=np.array(ok), **out)
    print(f'{len(ok)} converged coarse oracle runs ({len(names) - len(ok)} skipped), '
          f'{len(out["run"])} rows -> {outFile}')


# ============= Oracles on the 4-layer mesh (sweep/oracle.py) ========================
nProc = 6                  # buildOracle processes
oracleFiles = {'unsteady': os.path.join(common.DATA, 'datasets', 'coarse_unsteady.npz'),
               'steady3d': os.path.join(common.DATA, 'datasets', 'coarse_steady3d.npz')}


def columnGeometry(geo, nodes, tets, xy, nds, cols, nDisk=400):
    """Geometry of the interior 3D cells of each column (cols (nc, 2) grid points), from
    the cylinder (geo) on the extruded coarse mesh: volFrac (solid volume / cell volume),
    N = sum_s frac_s (n ds)_s / D (outward area normal of the surface in the cell per
    unit span / D), r = body-piece centroid minus grid point [m]. xy, nds: the surface
    stations and their tent-weighted normals x arc length."""
    from learn import pilot3d
    from sweep import oracle
    thick = oracle.nLayers * oracle.hz
    level = np.rint(nodes[:, 2] / oracle.hz).astype(int)
    D = geo['D']
    u = (np.arange(nDisk) + 0.5) / nDisk - 0.5
    X, Y = np.meshgrid(geo['xc'] + D * u, geo['yc'] + D * u)
    disk = np.column_stack([X.ravel(), Y.ravel()])
    disk = disk[np.hypot(disk[:, 0] - geo['xc'], disk[:, 1] - geo['yc']) < D / 2]
    Ld = pilot3d.lengthsInCells(disk, nodes, tets, thick, oracle.nLayers * 16)   # (nd, nn)
    Ls = pilot3d.lengthsInCells(xy, nodes, tets, thick, oracle.nLayers * oracle.nzSample)
    V = pilot3d.dualVolumes(nodes, tets)
    dA = (D / nDisk) ** 2
    out = {k: np.full(len(cols), np.nan) for k in ('volFrac', 'Nx', 'Ny', 'rx', 'ry')}
    for c, (cx, cy) in enumerate(cols):
        at = np.flatnonzero((np.hypot(nodes[:, 0] - cx, nodes[:, 1] - cy) < 1e-6)
                            & (level > 0) & (level < oracle.nLayers))
        i = at[0]                                       # interior cells are identical
        w = Ld[:, i]
        out['volFrac'][c] = w.sum() * dA / V[i]
        N = Ls[:, i] @ nds / oracle.hz / D
        out['Nx'][c], out['Ny'][c] = N
        if w.sum() > 0:                                 # sliver: no solid volume -> NaN r
            out['rx'][c], out['ry'][c] = (w @ disk) / w.sum() - [cx, cy]
    return out


def buildOracle(kind):
    """One row per interior source node per snapshot of every finished coarse oracle run:
    the coarse state at the node, the column geometry, and the source per unit span
    S = f h^2 applied at that time (interpolated from the resolved snapshots exactly as
    AERO-F does). Same columns as build(), plus snap, level, t and group (one (run,
    snapshot, level) layer: Cd/Cl are per group).
      kind 'unsteady': unsteady/oracle/runs/<case> (DONE), every snapshot
      kind 'steady3d': unsteady/oracle/steady/<case> (DONE and converged), the one
                       final state, with the constant source of steady-v0/runs/<case>
    Cases run in parallel (nProc processes): columnGeometry takes ~30 s per case."""
    from multiprocessing import Pool
    from sweep import oracle
    steady = kind == 'steady3d'
    orcRuns = oracle.steadyDir if steady else os.path.join(oracle.oracleDir, 'runs')
    names = sorted(n for n in os.listdir(orcRuns)
                   if os.path.isfile(os.path.join(orcRuns, n, 'DONE'))
                   and (not steady or converged(os.path.join(orcRuns, n))))
    with Pool(nProc) as pool:
        parts = pool.starmap(oracleCaseRows, [(kind, k, n) for k, n in enumerate(names)])
    out = {key: np.concatenate([np.array(p[key], dtype=float) for p in parts])
           for key in parts[0]}
    outFile = oracleFiles[kind]
    os.makedirs(os.path.dirname(outFile), exist_ok=True)
    np.savez(outFile, runNames=np.array(names), **out)
    print(f'{len(names)} oracle runs, {len(out["run"])} rows -> {outFile}')


def oracleCaseRows(kind, k, name):
    """buildOracle's rows (dict of lists) of oracle case name, run index k."""
    from learn import pilot3d
    from sweep import meshGen, oracle
    steady = kind == 'steady3d'
    orcRuns = oracle.steadyDir if steady else os.path.join(oracle.oracleDir, 'runs')
    resolved = os.path.join(common.STEADY if steady else common.UNSTEADY, 'runs')
    nodes, tets = common.readTop(oracle.meshFile)
    level = np.rint(nodes[:, 2] / oracle.hz).astype(int)
    gradient = leastSquaresGradient(nodes, tets)
    h = meshGen.h

    rows = {}
    add = lambda k, v: rows.setdefault(k, []).append(v)
    d = os.path.join(orcRuns, name)
    src = np.load(os.path.join(d, 'oracleSource.npz'))
    c = oracle.parseCase(name)
    mach, rhoInf, pInf = common.readInlet(os.path.join(d, 'FluidFile'))
    qInf = 0.5 * common.gamma * pInf * mach ** 2           # rho U^2 / 2
    geo = meshGen.loadGeometry(os.path.join(resolved, name))
    X = pilot3d.readSurfaceNodes(geo['surfFile'])
    xy = np.unique(np.round(X[:, :2], 12), axis=0)
    inner = (src['level'] > 0) & (src['level'] < oracle.nLayers)
    node, col = src['node'][inner], src['col'][inner]
    cols = np.array([nodes[node[col == j][0], :2] for j in range(col.max() + 1)])
    G = columnGeometry(geo, nodes, tets, xy, pilot3d.stationNormals(xy), cols)

    read = lambda q: common.readXpost(os.path.join(d, 'postpro', f'{q}.0000.xpost'))
    t, rho = read('Density')
    vel, p = read('Velocity')[1], read('Pressure')[1]
    S = src['f'][:, inner] * h ** 2                 # (nts, ni, 2) per unit span [N/m]
    for n in range(len(t)):
        V = np.column_stack([rho[n], vel[n, :, 0], vel[n, :, 1], p[n]])
        Gn = gradient(V)
        Sn = np.stack([[np.interp(t[n], src['t'], S[:, i, a]) for a in range(2)]
                       for i in range(len(node))])
        for i, (nd, j) in enumerate(zip(node, col)):
            for key, val in (('run', k), ('machInf', c.mach), ('ReInf', c.Re),
                             ('ox', c.ox), ('oy', c.oy),
                             ('qInf', qInf), ('duplicate', 0),
                             ('x', cols[j, 0]), ('y', cols[j, 1]), ('h', h),
                             ('D', geo['D']), ('Sx', Sn[i, 0]), ('Sy', Sn[i, 1]),
                             ('snap', n), ('level', level[nd]), ('t', t[n]),
                             ('group', (k * 10000 + n) * 10 + level[nd])):
                add(key, val)
            for key in ('volFrac', 'Nx', 'Ny', 'rx', 'ry'):
                add(key, G[key][j])
            for m, q in enumerate(stateNames):
                add(q, V[nd, m])
                add(f'd{q}_dx', Gn[nd, m, 0])
                add(f'd{q}_dy', Gn[nd, m, 1])
    print(f'{k}: {name}: {len(t)} snapshots x {len(node)} interior nodes', flush=True)
    return rows


if __name__ == '__main__':
    import sys
    if sys.argv[1:] and sys.argv[1] in oracleFiles:
        buildOracle(sys.argv[1])
    else:
        build()
