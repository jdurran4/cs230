"""A posteriori check of unsteady online runs against the coarse oracle (sweep/online.py,
sweep/oracle.py), for one case:

  Cd(t), Cl(t)   true: the oracle's source (the binned resolved force, as applied).
                 online: the network's source at the online run's snapshots, recomputed
                 with online/check.py (exactly what the C++ applies) at an interior level.
  field error    RMS over the interior-level nodes of the online - oracle state (p / p_inf,
                 u / U, v / U), divided by the RMS of the oracle's disturbance (oracle -
                 freestream): 0 = the oracle solution, 1 = an error as large as the flow
                 disturbance itself.

Usage (from the repo root):
  python -m online.unsteady CASE TAG [TAG ...]   (online runs unsteady/online/CASE_TAG)
Writes unsteady/online/CASE_compare.png and prints a table.
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common
from online import check
from sweep import meshGen, newRun, oracle

colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # categorical slots 1, 2, 3, 4
level = 2                                                # interior z-level used


def fields(d):
    """Times (nt,) and p, rho, u, v (nt, nn) of a run's postpro/*.0000.xpost."""
    read = lambda q: common.readXpost(os.path.join(d, 'postpro', f'{q}.0000.xpost'))
    t, p = read('Pressure')
    rho, vel = read('Density')[1], read('Velocity')[1]
    n = min(len(t), len(rho), len(vel))
    return t[:n], dict(p=p[:n], rho=rho[:n], u=vel[:n, :, 0], v=vel[:n, :, 1])


def main(name, tags):
    c = oracle.parseCase(name)
    fs = newRun.freestream(c.mach, c.Re)
    qD = 0.5 * fs['rho'] * fs['U'] ** 2 * meshGen.D
    nodes, _ = common.readTop(oracle.meshFile)
    lev = np.rint(nodes[:, 2] / oracle.hz).astype(int)
    on = np.flatnonzero(lev == level)
    at = lambda x, y: on[np.argmin(np.hypot(nodes[on, 0] - x, nodes[on, 1] - y))]

    # oracle: true source at its snapshots (interior cells, per unit span)
    od = os.path.join(oracle.oracleDir, 'runs', name)
    src = np.load(os.path.join(od, 'oracleSource.npz'))
    inner = src['level'] == level
    S = src['f'][:, inner] * meshGen.h ** 2                        # (nts, ncol, 2)
    tO, fO = fields(od)
    St = np.stack([[np.interp(tO, src['t'], S[:, j, a]) for a in range(2)]
                   for j in range(S.shape[1])])                      # (ncol, 2, nt)
    curves = {'oracle (true)': (tO, -St[:, 0].sum(0) / qD, -St[:, 1].sum(0) / qD, None)}

    for tag in tags:
        d = os.path.join(common.UNSTEADY, 'online', f'{name}_{tag}')
        weights = re.search(r'NeuralSourceWeightsFile\s*=\s*"([^"]+)"',
                            open(os.path.join(d, 'FluidFile')).read()).group(1)
        net = check.loadNet(os.path.join(d, weights))
        consts, cols = check.loadGeometry(os.path.join(d, 'nnGeometry.txt'))
        t, f = fields(d)
        node = [at(col[0], col[1]) for col in cols]
        Sn = np.array([[check.source(net, consts, col, f['rho'][n, i], f['u'][n, i],
                                     f['v'][n, i], f['p'][n, i])[2]
                        for col, i in zip(cols, node)] for n in range(len(t))])
        k = np.array([np.argmin(np.abs(tO - tt)) for tt in t])        # matching oracle snapshot
        err = {}
        for q, ref, free in (('p', fs['p'], fs['p']), ('u', fs['U'], fs['U']), ('v', fs['U'], 0.0)):
            e = np.sqrt(np.mean((f[q][:, on] - fO[q][k][:, on]) ** 2, axis=1)) / ref
            dist = np.sqrt(np.mean((fO[q][k][:, on] - free) ** 2, axis=1)) / ref
            err[q] = e / np.maximum(dist, 1e-12)
        curves[tag] = (t, -Sn[:, :, 0].sum(1) / qD, -Sn[:, :, 1].sum(1) / qD, err)

    # table
    print(f'{name}: Cd, Cl at t = 5, 10, 25, end (D/U); field error / disturbance (p, u, v) at the end')
    for lbl, (t, Cd, Cl, err) in curves.items():
        tDU = t / fs['tRef']
        pick = [np.argmin(np.abs(tDU - T)) for T in (5, 10, 25)] + [len(t) - 1]
        line = '  '.join(f'{Cd[i]:6.3f} {Cl[i]:+6.3f}' for i in pick)
        extra = '' if err is None else '   err ' + ' '.join(f'{err[q][-1]:.2f}' for q in 'puv')
        print(f'  {lbl:<16} {line}   (t_end = {tDU[-1]:.1f}){extra}')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for m, (lbl, (t, Cd, Cl, err)) in enumerate(curves.items()):
        col = '#52514e' if err is None else colors[(m - 1) % len(colors)]
        ax[0].plot(t / fs['tRef'], Cd, color=col, lw=2, label=lbl)
        ax[1].plot(t / fs['tRef'], Cl, color=col, lw=2, label=lbl)
        if err is not None:
            ax[2].plot(t / fs['tRef'], err['p'], color=col, lw=2, label=f'{lbl}')
    for a, yl in zip(ax, ('Cd', 'Cl', 'p error / p disturbance (vs oracle)')):
        a.set_xlabel('t  [D/U]')
        a.set_ylabel(yl)
        a.grid(color='#e1e0d9', lw=0.5)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    ax[0].legend(frameon=False)
    ax[2].set_ylim(bottom=0)
    fig.suptitle(f'{name}: online neural source vs coarse oracle')
    fig.tight_layout()
    out = os.path.join(common.UNSTEADY, 'online', f'{name}_compare.png')
    fig.savefig(out, dpi=120)
    print('wrote', out)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
