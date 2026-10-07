"""Write the oracle source file for a coarse AERO-F run (Equations/OracleSourceFile).

The oracle is the exact source S binned from a resolved run onto the coarse simulation
grid (grid 0 of samples.npz, from binMulti.py): the force the cylinder exerts on the
fluid in each coarse column, per unit span [N/m]. AERO-F needs a force per unit volume,
so each column gets the force density f = S / h^2 [N/m^3] (the column's dual cell is
h x h in x-y). AERO-F multiplies f by each node's control volume, which splits the force
between the column's two z-levels.

File format (plain numbers, no header, dimensional SI units), one line per column:
    x  y  fx  fy
x, y are the coarse node coordinates (AERO-F finds each node's line by matching them).

Usage:  python oracleSource.py runs/<case> [out file]
        (default out file: oracle/<case>/oracleSource.txt)
"""
import os
import sys

import numpy as np

run = sys.argv[1].rstrip('/')
case = os.path.basename(run)
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join('oracle', case, 'oracleSource.txt')

d = np.load(os.path.join(run, 'samples.npz'))
g0 = d['grid'] == 0                       # the simulation grid (D/h = 1.2, no offset)
x, y, h = d['colX'][g0], d['colY'][g0], d['h'][g0]
S = np.column_stack([d['Sx'][g0], d['Sy'][g0]])     # force on the fluid per span [N/m]
f = S / (h * h)[:, None]                  # force density [N/m^3]

os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
np.savetxt(out, np.column_stack([x, y, f]), fmt='%.16e')

qInf, D = float(d['qInf'][0]), float(d['D'][0])
print(f'{case}: {len(x)} columns -> {out}')
print(f'  total S = ({S[:, 0].sum():.6g}, {S[:, 1].sum():.6g}) N/m,  '
      f'Cd = {-S[:, 0].sum() / (qInf * D):.4f}, Cl = {-S[:, 1].sum() / (qInf * D):.4f}')
