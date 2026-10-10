"""Network inputs and targets from the raw dataset (learn/dataset.py).

Two frames for the target:
  'flow'  steady-v0: components along e = u/|u| and ePerp (ill-defined where u -> 0)
  'xy'    the solver's Cartesian frame
In both, the network predicts the source minus the pressure baseline pbar D N (the
pressure force on the column's piece of surface), normalized by (pbar + qbar) D:
  Y = (S - pbar D N) / ((pbar + qbar) D),    pbar = p, qbar = rho |u|^2 / 2 of the node.

Candidate inputs (all dimensionless):
  flow frame (steady-v0):  Mbar = |u|/c, logReD = log(rho |u| D / mu), Npar, Nperp = N.e,
                           N.ePerp, cosT, sinT = direction of r in the flow frame,
                           dh = |r|/h
  x-y frame:               Mx, My = u/c, v/c, absM = |u|/c, logRec = log(rho c D / mu),
                           Nx, Ny, rxh, ryh = r/h
  both:                    volFrac, Dh = D/h
  gradients (x-y):         h grad(rho)/rho, h grad(u)/c, h grad(v)/c, h grad(p)/p: the
                           change across one coarse cell, e.g. gx_u = h du/dx / c
"""
import numpy as np

import common

inputSets = {
    'flow (steady-v0)': (['Mbar', 'logReD', 'volFrac', 'Npar', 'Nperp', 'cosT', 'sinT', 'dh',
                          'Dh'], 'flow'),
    'xy':               (['Mx', 'My', 'logRec', 'volFrac', 'Nx', 'Ny', 'rxh', 'ryh', 'Dh'],
                         'xy'),
    'xy+absM':          (['Mx', 'My', 'absM', 'logRec', 'volFrac', 'Nx', 'Ny', 'rxh', 'ryh',
                          'Dh'], 'xy'),
}
gradNames = [f'g{a}_{q}' for q in ('rho', 'u', 'v', 'p') for a in 'xy']
inputSets['xy+grad'] = (inputSets['xy'][0] + gradNames, 'xy')
inputSets['xy+absM+grad'] = (inputSets['xy+absM'][0] + gradNames, 'xy')


# inputs that change sign when the flow is mirrored in y (y -> -y); the target's y
# component flips too. Gradients: d/dy flips, and v flips, so gy_v is unchanged.
mirrorFlips = {'My', 'Ny', 'ryh', 'gy_rho', 'gy_u', 'gx_v', 'gy_p'}


def mirrorSigns(names):
    """+-1 per input: multiply a row by it to get the mirror-image row (x-y sets only)."""
    return np.array([-1.0 if n in mirrorFlips else 1.0 for n in names])


def candidates(d):
    """Every candidate input, as a dict of arrays (one entry per row)."""
    rho, u, v, p, h, D = d['rho'], d['u'], d['v'], d['p'], d['h'], d['D']
    c = np.sqrt(common.gamma * p / rho)
    speed = np.hypot(u, v)
    mu = common.viscosity(p / (rho * common.R_gas))
    e, eP = flowFrame(d)
    N = np.column_stack([d['Nx'], d['Ny']])
    r = np.column_stack([d['rx'], d['ry']])
    rn = np.linalg.norm(r, axis=1)
    dot = lambda a, b: (a * b).sum(axis=1)
    f = dict(
        Mbar=speed / c, logReD=np.log(rho * speed * D / mu),
        Npar=dot(N, e), Nperp=dot(N, eP), cosT=dot(r, e) / rn, sinT=dot(r, eP) / rn, dh=rn / h,
        Mx=u / c, My=v / c, absM=speed / c, logRec=np.log(rho * c * D / mu),
        Nx=d['Nx'], Ny=d['Ny'], rxh=d['rx'] / h, ryh=d['ry'] / h,
        volFrac=d['volFrac'], Dh=D / h)
    for a in 'xy':
        f[f'g{a}_rho'] = h * d[f'drho_d{a}'] / rho
        f[f'g{a}_u'] = h * d[f'du_d{a}'] / c
        f[f'g{a}_v'] = h * d[f'dv_d{a}'] / c
        f[f'g{a}_p'] = h * d[f'dp_d{a}'] / p
    return f


def flowFrame(d):
    speed = np.hypot(d['u'], d['v'])
    e = np.column_stack([d['u'], d['v']]) / speed[:, None]
    return e, np.column_stack([-e[:, 1], e[:, 0]])


def scales(d):
    """pbar D N (the pressure baseline, N/m, x-y) and (pbar + qbar) D per row."""
    pB, qB = d['p'], 0.5 * d['rho'] * (d['u'] ** 2 + d['v'] ** 2)
    N = np.column_stack([d['Nx'], d['Ny']])
    return (pB * d['D'])[:, None] * N, (pB + qB) * d['D']


def target(d, frame):
    """Y (n, 2): the network's target in the given frame."""
    base, scale = scales(d)
    Sp = (np.column_stack([d['Sx'], d['Sy']]) - base) / scale[:, None]
    if frame == 'xy':
        return Sp
    e, eP = flowFrame(d)
    return np.column_stack([(Sp * e).sum(axis=1), (Sp * eP).sum(axis=1)])


def toSource(Y, d, frame):
    """Inverse of target: predictions Y (n, 2) -> S (n, 2) in N/m, x-y."""
    base, scale = scales(d)
    if frame == 'flow':
        e, eP = flowFrame(d)
        Y = Y[:, :1] * e + Y[:, 1:] * eP
    return Y * scale[:, None] + base
