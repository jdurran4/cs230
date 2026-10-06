"""
Phase 4 dataset: one row per (run, force cell) with every candidate NN input and target,
so input sets and normalizations can be compared without re-binning.

Reads runs/<run>/inputs.npz and targets.npz (run binInputs.py, then binTargets.py) and
keeps the columns the body touches (the target columns), at the last output step (steady).

Columns (all per row):
  meta     run (index into runNames), machInf, ReInf, ox, oy, zOverhang, colX, colY,
           xc, yc, h, D, qInf, duplicate (1 = _me2M cap test, a near copy of its 1M run)
  inputs   (local, what a coarse solver has online)
           Mbar, ReD, volFrac            cell-averaged Mach, Re_D, body volume fraction
           Npar, Nperp, Nmag             body normal N (per unit span / D) in the flow frame
           cosT, sinT                    theta = angle from ubar to r, r = body-piece
                                         centroid - grid point (sinT > 0: body left of ubar)
           dh, rpar, rperp               |r| / h and r in the flow frame / h
           Dh                            D / h (constant in this sweep)
           pqRatio                       pbar / qbar = 2 / (gamma Mbar^2), for reference
  targets  SxInf, SyInf                  S / (q_inf D), x-y frame          (freestream norm)
           Spar_q, Sperp_q               Sp . e / (qbar D)                 (Sp = S - pbar D N)
           Spar_pq, Sperp_pq             Sp . e / ((pbar + qbar) D)
           SFpar_q, SFperp_q             S . e / (qbar D)                  (full S, no pbar term)
           SFpar_pq, SFperp_pq           S . e / ((pbar + qbar) D)
  raw      Sx, Sy [N/m], pBar, qBar [Pa], ePar_x, ePar_y        to rebuild S from any target

Writes aerof/dataset.npz (arrays + runNames + featureNames) and aerof/dataset.csv.
"""
import os
import re

import numpy as np

import newRun

aerofDir = os.path.dirname(os.path.abspath(__file__))
runsDir = os.path.join(aerofDir, 'runs')
nameRe = re.compile(r'M([\d.]+)_Re([\d.]+)_x([\d.]+)_y([\d.]+)(?:_(.*))?$')


def rowsForRun(runName):
    d = os.path.join(runsDir, runName)
    inp, tg = np.load(os.path.join(d, 'inputs.npz')), np.load(os.path.join(d, 'targets.npz'))
    m = nameRe.match(runName)
    machInf, ReInf, ox, oy, tag = float(m[1]), float(m[2]), float(m[3]), float(m[4]), m[5] or ''
    geo = __import__('cylMeshGen').loadGeometry(d)
    D, h = float(tg['D']), geo['h']

    cols = tg['columns']
    ic = np.array([np.argmin(np.linalg.norm(inp['columns'] - c, axis=1)) for c in cols])
    assert np.allclose(inp['columns'][ic], cols)
    ti = np.argmin(np.abs(inp['times'] - tg['times'][-1]))
    t = -1                                                       # last (converged) step

    e = tg['ePar'][t]                                            # (ncol, 2)
    ePerp = np.column_stack([-e[:, 1], e[:, 0]])
    pBar, qBar = tg['pBar'][t], tg['qBar'][t]
    S, Sp, N = tg['S'][t], tg['Sp'][t], tg['N']
    r = inp['solidCentroid'][ic] - cols
    noSolid = inp['Vsolid'][ic] == 0
    r[noSolid] = np.nan                                          # surface-only sliver cells
    rn = np.linalg.norm(r, axis=1)
    dot = lambda a, b: (a * b).sum(axis=1)

    n = len(cols)
    full = lambda v: np.full(n, v, dtype=float)
    return dict(
        run=full(-1), machInf=full(machInf), ReInf=full(ReInf), ox=full(ox), oy=full(oy),
        zOverhang=full(geo['zOverhang']), colX=cols[:, 0], colY=cols[:, 1],
        xc=full(geo['xc']), yc=full(geo['yc']), h=full(h), D=full(D), qInf=full(float(tg['q'])),
        duplicate=full(tag == 'me2M'),
        Mbar=inp['M'][ti, ic], ReD=inp['ReD'][ti, ic], volFrac=inp['volFrac'][ic],
        Npar=dot(N, e), Nperp=dot(N, ePerp), Nmag=np.linalg.norm(N, axis=1),
        cosT=dot(r, e) / rn, sinT=dot(r, ePerp) / rn, dh=rn / h,
        rpar=dot(r, e) / h, rperp=dot(r, ePerp) / h, Dh=full(D / h),
        pqRatio=pBar / qBar,
        SxInf=S[:, 0] / (tg['q'] * D), SyInf=S[:, 1] / (tg['q'] * D),
        Spar_q=dot(Sp, e) / (qBar * D), Sperp_q=dot(Sp, ePerp) / (qBar * D),
        Spar_pq=dot(Sp, e) / ((pBar + qBar) * D), Sperp_pq=dot(Sp, ePerp) / ((pBar + qBar) * D),
        SFpar_q=dot(S, e) / (qBar * D), SFperp_q=dot(S, ePerp) / (qBar * D),
        SFpar_pq=dot(S, e) / ((pBar + qBar) * D), SFperp_pq=dot(S, ePerp) / ((pBar + qBar) * D),
        Sx=S[:, 0], Sy=S[:, 1], pBar=pBar, qBar=qBar, ePar_x=e[:, 0], ePar_y=e[:, 1])


if __name__ == '__main__':
    runNames = sorted(r for r in os.listdir(runsDir)
                      if os.path.isfile(os.path.join(runsDir, r, 'targets.npz')))
    parts = []
    for k, r in enumerate(runNames):
        rows = rowsForRun(r)
        rows['run'][:] = k
        parts.append(rows)
    keys = list(parts[0])
    data = {key: np.concatenate([p[key] for p in parts]) for key in keys}

    np.savez(os.path.join(aerofDir, 'dataset.npz'), runNames=np.array(runNames),
             featureNames=np.array(keys), **data)
    np.savetxt(os.path.join(aerofDir, 'dataset.csv'), np.column_stack([data[k] for k in keys]),
               delimiter=',', header=','.join(keys), comments='', fmt='%.8g')

    # summary
    keep = data['duplicate'] == 0
    print(f'{len(runNames)} runs, {len(data["run"])} rows ({keep.sum()} without the me2M '
          f'duplicates), {len(keys)} columns')
    print(f'cells with no body volume (theta, d/h = NaN): {np.isnan(data["dh"]).sum()}')
    print(f'{"column":>10} {"min":>10} {"median":>10} {"max":>10}')
    for k in keys[keys.index('Mbar'):]:
        v = data[k][keep]
        print(f'{k:>10} {np.nanmin(v):10.3g} {np.nanmedian(v):10.3g} {np.nanmax(v):10.3g}')
    # per-run consistency: sum of cell sources = -Cd
    cd = np.array([-data['SxInf'][data['run'] == k].sum() for k in range(len(runNames))])
    for k, r in enumerate(runNames):
        print(f'   {r:<32} cells {np.sum(data["run"] == k):2d}  Cd {cd[k]:.3f}')
