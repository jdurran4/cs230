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

Usage (from the repo root):  python -m learn.dataset   -> DATA/datasets/coarse_steady.npz
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


if __name__ == '__main__':
    build()
