"""Python reference of AERO-F's neural source (VolumicForceTerm::addNeuralSource).

source() does exactly what the C++ does at one node: node state (rho, u, v, p) [SI] +
column geometry -> x-y inputs -> network -> force on the fluid per unit span S [N/m]:
  inputs  Mx = u/c, My = v/c, logRec = log(rho c D / mu(T)), volFrac, Nx, Ny, rx/h, ry/h,
          D/h                                             (learn/features.py, set 'xy')
  S       = (pbar + qbar) D out + (pbar - pRef) D N,   pbar = p, qbar = rho |u|^2 / 2
            pRef = p_inf for the p - p_inf nets, 0 for the absolute-pressure ones
The C++ then adds -ctrlVol * S / h^2 (nondimensionalized) to the momentum residual.

Running it on a case prints the C++ test values: inputs, outputs, and S of column 0 at
the freestream state (iteration 0 of an online run), and the total Cd, Cl the network
predicts at freestream.

Usage (from the repo root):
  python -m online.check NETDIR CASEDIR
    NETDIR   has nnWeights.txt and nnTest.txt (learn/train.py export)
    CASEDIR  has nnGeometry.txt and the FluidFile (freestream from its Inlet block)
"""
import os
import sys

import numpy as np

import common

inputTag = 'xy'            # the input set this reference implements
clamp = 4.0                # standardized inputs are clipped to +-clamp (nnClamp in the C++)


def loadNet(path):
    """The network in nnWeights.txt (format in learn/train.py), as a function x -> y."""
    vals = iter(open(path).read().split())
    tag = next(vals)
    assert tag == inputTag, f'{path} is for input set {tag!r}, not {inputTag!r}'
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
        a = np.clip((x - xMean) / xStd, -clamp, clamp)    # no extrapolation past the clamp
        for l in range(L):
            a = Ws[l] @ a + bs[l]
            if l < L - 1:
                a = np.tanh(a)
        return a * yStd + yMean
    return net


def loadGeometry(path):
    """Gas constants and pRef (first line; older files have no pRef) and one row per
    source column: x y h D volFrac Nx Ny rx ry (N and r already in x-y)."""
    with open(path) as f:
        consts = tuple(map(float, f.readline().split()))
        cols = np.array([list(map(float, line.split())) for line in f if line.strip()])
    return consts, cols


def source(net, consts, col, rho, u, v, p):
    """(inputs, outputs, S [N/m]) at one node, as the C++ computes them."""
    gamma, R, C, Sref, *rest = consts
    pRef = rest[0] if rest else 0.0
    x, y, h, D, volFrac, Nx, Ny, rx, ry = col
    c = np.sqrt(gamma * p / rho)
    T = p / (rho * R)
    mu = C * T ** 1.5 / (T + Sref)
    inputs = np.array([u / c, v / c, np.log(rho * c * D / mu), volFrac, Nx, Ny, rx / h,
                       ry / h, D / h])
    out = net(inputs)
    pBar, qBar = p, 0.5 * rho * (u * u + v * v)
    S = (pBar + qBar) * D * out + (pBar - pRef) * D * np.array([Nx, Ny])
    return inputs, out, S


if __name__ == '__main__':
    netDir, caseDir = sys.argv[1:3]
    net = loadNet(os.path.join(netDir, 'nnWeights.txt'))
    for line in open(os.path.join(netDir, 'nnTest.txt')):
        if not line.startswith('#'):
            xs, ys = line.split('->')
            err = np.abs(net(np.array(xs.split(), float)) - np.array(ys.split(), float)).max()
            assert err < 1e-8, f'text-file network differs from the trained one: {err:.1e}'
    print('nnWeights.txt reproduces nnTest.txt')

    consts, cols = loadGeometry(os.path.join(caseDir, 'nnGeometry.txt'))
    assert np.allclose(consts[:4], (common.gamma, common.R_gas, common.C_suth, common.S_suth))
    mach, rho, p = common.readInlet(os.path.join(caseDir, 'FluidFile'))
    u = mach * np.sqrt(common.gamma * p / rho)
    print(f'freestream rho u v p = {rho:.10g} {u:.10g} 0 {p:.10g}')

    total = np.zeros(2)
    for k, col in enumerate(cols):
        inputs, out, S = source(net, consts, col, rho, u, 0.0, p)
        total += S
        if k == 0:
            print('column 0 (C++ test values):')
            print('  inputs ', ' '.join(f'{a:.6g}' for a in inputs))
            print('  outputs', ' '.join(f'{a:.6g}' for a in out))
            print(f'  S = {S[0]:.8g} {S[1]:.8g} [N/m]')
    qD = 0.5 * rho * u * u * cols[0, 3]
    print(f'{len(cols)} columns at freestream: Cd {-total[0] / qD:.3f}, Cl {-total[1] / qD:+.3f}')
