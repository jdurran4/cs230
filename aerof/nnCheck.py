"""Reference implementation of the online NN source, and the offline shift check.

source(state, geo) does exactly what AERO-F will do at one node: node state (rho, u, v,
p) [SI] + column geometry -> NN inputs -> NN -> force on the fluid per unit span S [N/m].
The C++ version must reproduce it (step E test: freestream state, iteration 0).

Shift check: the network was trained on fine-solution averages over each column. Online
it sees the coarse solver's node state instead. Evaluate it on
  fine    the training inputs (fine averages)     -> the NN's own error
  oracle  the converged oracle run's node states  -> the distribution shift at the
          state the online run should approach
  inf     freestream (iteration 0 of an online run)
and compare S with the true binned S.

Usage:  python nnCheck.py <case>     (needs nn/, online/<case>/nnGeometry.txt,
                                      oracle/<case>/postpro from the oracle run)
"""
import os
import sys

import numpy as np
import torch

import binOneCell as bin1
from binInputs import readXpost
import nnExport

here = os.path.dirname(os.path.abspath(__file__))


def loadModel():
    """Rebuild the network from nnWeights.txt (the file AERO-F reads)."""
    vals = iter(np.loadtxt(os.path.join(here, 'nn', 'nnWeights.txt'), comments='#',
                           converters=None, ndmin=1).ravel()
                if False else open(os.path.join(here, 'nn', 'nnWeights.txt')).read().split())
    nxt = lambda: float(next(vals))
    L = int(nxt())
    sizes = [int(nxt()) for _ in range(L + 1)]
    xMean = np.array([nxt() for _ in range(sizes[0])])
    xStd = np.array([nxt() for _ in range(sizes[0])])
    Ws, bs = [], []
    for l in range(L):
        Ws.append(np.array([[nxt() for _ in range(sizes[l])] for _ in range(sizes[l + 1])]))
        bs.append(np.array([nxt() for _ in range(sizes[l + 1])]))
    yMean = np.array([nxt() for _ in range(sizes[-1])])
    yStd = np.array([nxt() for _ in range(sizes[-1])])

    def net(x):
        a = (x - xMean) / xStd
        for l in range(L):
            a = Ws[l] @ a + bs[l]
            if l < L - 1:
                a = np.tanh(a)
        return a * yStd + yMean
    return net


def loadGeometry(path):
    with open(path) as f:
        gamma, R, C, Sref = map(float, f.readline().split())
        cols = np.array([list(map(float, line.split())) for line in f if line.strip()])
    return (gamma, R, C, Sref), cols


def source(net, consts, col, rho, u, v, p, verbose=False):
    """Force on the fluid per unit span [N/m] from one node state, as AERO-F computes it."""
    gamma, R, C, Sref = consts
    x, y, h, D, volFrac, Nx, Ny, rx, ry = col
    speed = np.hypot(u, v)
    e = np.array([u, v]) / speed
    ePerp = np.array([-e[1], e[0]])
    T = p / (rho * R)
    mu = C * T ** 1.5 / (T + Sref)
    N, r = np.array([Nx, Ny]), np.array([rx, ry])
    rn = np.linalg.norm(r)
    inputs = np.array([speed / np.sqrt(gamma * p / rho), np.log(rho * speed * D / mu), volFrac,
                       N @ e, N @ ePerp, r @ e / rn, r @ ePerp / rn, rn / h, D / h])
    out = net(inputs)
    pBar, qBar = p, 0.5 * rho * speed ** 2
    S = (pBar + qBar) * D * (out[0] * e + out[1] * ePerp) + pBar * D * N
    if verbose:
        print('    inputs', ' '.join(f'{a:.6g}' for a in inputs))
        print('    outputs', ' '.join(f'{a:.6g}' for a in out), ' S', S)
    return S


if __name__ == '__main__':
    case = sys.argv[1]
    net = loadModel()
    # sanity: the text-file network reproduces nnTest.txt
    for line in open(os.path.join(here, 'nn', 'nnTest.txt')).readlines()[1:]:
        xs, ys = line.split('->')
        err = np.abs(net(np.array(xs.split(), float)) - np.array(ys.split(), float)).max()
        assert err < 1e-8, err
    consts, cols = loadGeometry(os.path.join(here, 'online', case, 'nnGeometry.txt'))

    # true S and the fine (training) inputs on grid 0
    smp = np.load(os.path.join(here, 'runs', case, 'samples.npz'))
    g0 = smp['grid'] == 0
    fine = np.load(os.path.join(here, 'runs', case, 'inputs.npz'))
    fineCols = fine['columns']

    # oracle run: node states, mean of the 2 z-levels per column
    post = os.path.join(here, 'oracle', case, 'postpro')
    nodes, _ = bin1.readDomain(os.path.join(post, 'OUTPUT.top.0000'))
    xyC, colOf = np.unique(np.round(nodes[:, :2], 9), axis=0, return_inverse=True)
    colOf = colOf.ravel()
    cm = lambda a: np.bincount(colOf, a) / np.bincount(colOf)
    vel = readXpost(os.path.join(post, 'Velocity.0000.xpost'))[1][-1]
    orc = dict(rho=cm(readXpost(os.path.join(post, 'Density.0000.xpost'))[1][-1]),
               u=cm(vel[:, 0]), v=cm(vel[:, 1]),
               p=cm(readXpost(os.path.join(post, 'Pressure.0000.xpost'))[1][-1]))

    # freestream from the fine run's upstream-most column
    i0 = np.argmin(fineCols[:, 0] + 1e-3 * np.abs(fineCols[:, 1]))
    inf = dict(rho=fine['rho'][-1, i0], u=fine['u'][-1, i0, 0], v=0.0, p=fine['p'][-1, i0])
    qInf, D = 0.5 * inf['rho'] * inf['u'] ** 2, cols[0, 3]

    tot = {k: np.zeros(2) for k in ('true', 'fine', 'oracle', 'inf')}
    print(f'{case}: S / (q_inf D) per source column  [x, y components]')
    print(f'{"x":>7s} {"y":>6s} | {"true":>15s} | {"NN(fine)":>15s} | {"NN(oracle)":>15s} | '
          f'{"NN(freestream)":>15s} | Mbar fine/oracle  ReD fine/oracle')
    for col in cols:
        x, y = col[:2]
        k = np.argmin(np.hypot(smp['colX'] - x, smp['colY'] - y) + 1e9 * ~g0)
        j = np.argmin(np.hypot(fineCols[:, 0] - x, fineCols[:, 1] - y))
        c = np.argmin(np.hypot(xyC[:, 0] - x, xyC[:, 1] - y))
        Strue = np.array([smp['Sx'][k], smp['Sy'][k]])
        Sf = source(net, consts, col, fine['rho'][-1, j], *fine['u'][-1, j], fine['p'][-1, j])
        So = source(net, consts, col, orc['rho'][c], orc['u'][c], orc['v'][c], orc['p'][c])
        Si = source(net, consts, col, inf['rho'], inf['u'], inf['v'], inf['p'])
        for name, S in zip(tot, (Strue, Sf, So, Si)):
            tot[name] += S
        mo = np.hypot(orc['u'][c], orc['v'][c]) / np.sqrt(1.4 * orc['p'][c] / orc['rho'][c])
        f2 = lambda S: f'{S[0] / (qInf * D):+7.3f} {S[1] / (qInf * D):+7.3f}'
        print(f'{x:7.2f} {y:6.2f} | {f2(Strue)} | {f2(Sf)} | {f2(So)} | {f2(Si)} | '
              f'{fine["M"][-1, j]:.2f} / {mo:.2f}   {fine["ReD"][-1, j]:5.1f} / '
              f'{orc["rho"][c] * np.hypot(orc["u"][c], orc["v"][c]) * D / (consts[2] * (orc["p"][c] / (orc["rho"][c] * consts[1])) ** 1.5 / (orc["p"][c] / (orc["rho"][c] * consts[1]) + consts[3])):5.1f}')
    print('total Cd, Cl (= -sum S / q_inf D):')
    for name, S in tot.items():
        print(f'  {name:8s} Cd {-S[0] / (qInf * D):6.3f}   Cl {-S[1] / (qInf * D):+6.3f}')
    print('step E test (freestream, first source column):')
    source(net, consts, cols[0], inf['rho'], inf['u'], inf['v'], inf['p'], verbose=True)
    print(f'    freestream rho u p = {inf["rho"]:.10g} {inf["u"]:.10g} {inf["p"]:.10g}')
