"""Compare a coarse oracle run with the resolved run binned onto the same coarse grid.

The oracle run (oracle/<case>, coarse mesh, no cylinder, exact binned source) gives one
state per coarse node; a column's value is the mean of its two z-level nodes. The
reference is the resolved run's fluid-part average over each column (inputs.npz from
binInputs.py). A uniform freestream field (= the coarse run with no source) is the
"no model" baseline.

Score per field:  ratio = RMS(oracle - fine) / RMS(freestream - fine)
    0 = the oracle run reproduces the binned resolved flow, 1 = no better than ignoring
    the body. Columns with volFrac > 0 (the body's own columns) are reported separately.

Usage:  python oracleCompare.py <case> [base]   (e.g. M1.5_Re100_x0.04_y0.12)
    base: the directory holding runs/ and oracle/ (default: aerof/; dh025 for that test)
Writes <base>/oracle/<case>/compare.png
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import binOneCell as bin1
from binInputs import readXpost

g = 1.4
case = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else '.'
oracleDir = os.path.join(base, 'oracle', case)
post = os.path.join(oracleDir, 'postpro')

# ---- coarse oracle solution, averaged over each column's two z-level nodes ----------
nodes, _ = bin1.readDomain(os.path.join(post, 'OUTPUT.top.0000'))
columns, colOf = np.unique(np.round(nodes[:, :2], 9), axis=0, return_inverse=True)
colOf = colOf.ravel()
nCol = len(columns)
count = np.bincount(colOf, minlength=nCol)


def colMean(v):
    return np.bincount(colOf, weights=v, minlength=nCol) / count


rhoC = colMean(readXpost(os.path.join(post, 'Density.0000.xpost'))[1][-1])
velC = readXpost(os.path.join(post, 'Velocity.0000.xpost'))[1][-1]
uC, vC = colMean(velC[:, 0]), colMean(velC[:, 1])
pC = colMean(readXpost(os.path.join(post, 'Pressure.0000.xpost'))[1][-1])

# ---- resolved run, binned (fluid-part averages) --------------------------------------
fine = np.load(os.path.join(base, 'runs', case, 'inputs.npz'))
assert np.allclose(fine['columns'], columns, atol=1e-3), 'column order differs from inputs.npz'
rhoF, uF, vF, pF = fine['rho'][-1], fine['u'][-1, :, 0], fine['u'][-1, :, 1], fine['p'][-1]
volFrac = fine['volFrac']

# ---- freestream baseline (upstream-most column of the fine run) ----------------------
iInf = np.argmin(columns[:, 0] + 1e-3 * np.abs(columns[:, 1]))
inf = dict(rho=rhoF[iInf], u=uF[iInf], v=0.0, p=pF[iInf])
Uinf = np.hypot(inf['u'], inf['v'])
cInf = np.sqrt(g * inf['p'] / inf['rho'])

fields = {  # name: (oracle, fine, freestream, scale for plots)
    'rho / rho_inf': (rhoC / inf['rho'], rhoF / inf['rho'], 1.0),
    'u / U_inf':     (uC / Uinf, uF / Uinf, 1.0),
    'v / U_inf':     (vC / Uinf, vF / Uinf, 0.0),
    'p / p_inf':     (pC / inf['p'], pF / inf['p'], 1.0),
    'Mach':          (np.hypot(uC, vC) / np.sqrt(g * pC / rhoC),
                      np.hypot(uF, vF) / np.sqrt(g * pF / rhoF), Uinf / cInf),
}

body = volFrac > 0
rms = lambda a: np.sqrt(np.mean(a ** 2))
print(f'{case}: {nCol} columns ({body.sum()} with body volume)')
print(f'{"field":15s} {"RMS(oracle-fine)":>17s} {"RMS(inf-fine)":>14s} {"ratio":>7s}'
      f'   {"ratio off-body":>14s}')
for name, (o, f, fInf) in fields.items():
    r = rms(o - f) / rms(fInf - f)
    rOff = rms((o - f)[~body]) / rms((fInf - f)[~body])
    print(f'{name:15s} {rms(o - f):17.4f} {rms(fInf - f):14.4f} {r:7.3f}   {rOff:14.3f}')

# ---- plots: fine binned | oracle | difference, per field -----------------------------
x, y = fine['columns'][:, 0], fine['columns'][:, 1]
fig, axes = plt.subplots(len(fields), 3, figsize=(15, 3.2 * len(fields)), squeeze=False)
for row, (name, (o, f, fInf)) in zip(axes, fields.items()):
    lo, hi = min(o.min(), f.min()), max(o.max(), f.max())
    d = np.abs(o - f).max()
    for ax, vals, title, kw in [
            (row[0], f, f'resolved, binned: {name}', dict(vmin=lo, vmax=hi, cmap='viridis')),
            (row[1], o, f'coarse + oracle source: {name}', dict(vmin=lo, vmax=hi, cmap='viridis')),
            (row[2], o - f, 'oracle - resolved', dict(vmin=-d, vmax=d, cmap='RdBu_r'))]:
        sc = ax.tricontourf(x, y, vals, levels=40, **kw)
        ax.plot(fine['xc'], fine['yc'], 'k+', ms=8)
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal')
        fig.colorbar(sc, ax=ax, shrink=0.8)
fig.tight_layout()
out = os.path.join(oracleDir, 'compare.png')
fig.savefig(out, dpi=110)
print('wrote', out)
