"""Train the deployment network and write the files AERO-F reads for the online source.

1. Train one MLP on ALL of dataset_multi.npz (no held-out folds: ablation.py already
   measured the generalization error of this configuration). Same settings as
   ablation.py: inputs base+theta+dh+Dh, target Sp_pq, 32 x 32 tanh, Adam, full batch.
2. Write nnWeights.txt (network) and, per case, nnGeometry.txt (fixed column geometry).
3. Write nnTest.txt: a few input rows and the network's outputs, to check the C++
   forward pass against.

Usage:  python nnExport.py train [dataset] [outDir]   -> <outDir>/nnWeights.txt, nnTest.txt
            (default dataset_multi.npz -> nn/; dataset_coarse.npz -> nn/coarse/ for the
             network trained on coarse oracle states, see coarseOracle.py)
        python nnExport.py geometry runs/<case> <outDir>   -> <outDir>/nnGeometry.txt

---- nnWeights.txt (plain numbers, whitespace separated, read in this order) ----------
    L                         number of linear layers (3)
    n0 n1 ... nL              layer widths (9 32 32 2)
    xMean[n0]  xStd[n0]       input standardization:  z = (x - xMean) / xStd
    for each layer l = 1..L:
        W[n_l][n_{l-1}]       row by row (row = one output neuron)
        b[n_l]
    yMean[nL]  yStd[nL]       output de-standardization:  y = out * yStd + yMean
  Forward pass:  a = z;  a = tanh(W1 a + b1);  a = tanh(W2 a + b2);  out = W3 a + b3
  Inputs, in order: Mbar logReD volFrac Npar Nperp cosT sinT dh Dh
  Outputs: Spar_pq Sperp_pq = (Sp . e, Sp . ePerp) / ((pbar + qbar) D)

---- nnGeometry.txt -------------------------------------------------------------------
    first line:  gamma R_gas C_suth S_suth      (gas constants used to build the dataset)
    then one line per source column:
        x y h D volFrac Nx Ny rx ry
  N = sum of the surface's outward area normals in the column / D (per unit span),
  r = (body-piece centroid - grid point) [m]. Online, with e = u / |u| from the node:
    Npar = N . e, Nperp = N . ePerp, cosT = r . e / |r|, sinT = r . ePerp / |r|,
    dh = |r| / h, Dh = D / h, Mbar = |u| / c, ReD = rho |u| D / mu(T)
  and the force on the fluid per unit span is
    S = (pbar + qbar) D (Spar_pq e + Sperp_pq ePerp) + pbar D N        [N/m]
  so the force density is S / h^2 [N/m^3] (as in the oracle file).
"""
import os
import sys

import numpy as np
import torch

import newRun

features = ['Mbar', 'logReD', 'volFrac', 'Npar', 'Nperp', 'cosT', 'sinT', 'dh', 'Dh']
targets = ['Spar_pq', 'Sperp_pq']
hidden = [32, 32]
nEpochs, lr, weightDecay, seed = 3000, 3e-3, 1e-4, 0
here = os.path.dirname(os.path.abspath(__file__))
nnDir = os.path.join(here, 'nn')


def loadData(dataset='dataset_multi.npz'):
    raw = np.load(os.path.join(here, dataset))
    d = {k: raw[k] for k in raw['featureNames']}
    d['logReD'] = np.log(d['ReD'])
    keep = (d['duplicate'] == 0) & np.all([np.isfinite(d[f]) for f in features + targets],
                                          axis=0)
    return {k: v[keep] for k, v in d.items()}


def train(dataset='dataset_multi.npz', nnDir=nnDir):
    d = loadData(dataset)
    X = np.column_stack([d[f] for f in features])
    Y = np.column_stack([d[t] for t in targets])
    xMean, xStd = X.mean(axis=0), X.std(axis=0)
    # a constant input (Dh on grid 0 only) has std ~1e-14 from rounding: dividing by it
    # would blow rounding differences up to O(1) inputs. Leave such inputs unscaled.
    xStd[xStd < 1e-8 * np.maximum(np.abs(xMean), 1.0)] = 1.0
    yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

    torch.manual_seed(seed)
    layers, width = [], len(features)
    for h in hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.Tanh()]
        width = h
    layers.append(torch.nn.Linear(width, len(targets)))
    model = torch.nn.Sequential(*layers).double()     # float64: C++ uses double
    Xt = torch.tensor((X - xMean) / xStd)
    Yt = torch.tensor((Y - yMean) / yStd)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
    for epoch in range(nEpochs):
        opt.zero_grad()
        loss = torch.mean((model(Xt) - Yt) ** 2)
        loss.backward()
        opt.step()
    print(f'{len(X)} rows, final training loss (standardized MSE) {loss.item():.4g}')

    os.makedirs(nnDir, exist_ok=True)
    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    with open(os.path.join(nnDir, 'nnWeights.txt'), 'w') as f:
        f.write(f'{len(linears)}\n')
        f.write(' '.join(str(n) for n in [len(features)] + hidden + [len(targets)]) + '\n')
        for v in (xMean, xStd):
            f.write(' '.join(f'{a:.17g}' for a in v) + '\n')
        for lin in linears:
            for row in lin.weight.detach().numpy():
                f.write(' '.join(f'{a:.17g}' for a in row) + '\n')
            f.write(' '.join(f'{a:.17g}' for a in lin.bias.detach().numpy()) + '\n')
        for v in (yMean, yStd):
            f.write(' '.join(f'{a:.17g}' for a in v) + '\n')

    # test rows: 3 training rows + a freestream-like input in a body column
    rows = [X[0], X[len(X) // 2], X[-1],
            np.array([1.5, np.log(100.0), 0.8, 0.1, -0.3, 0.06, 0.99, 0.08, 1.2])]
    with torch.no_grad():
        out = model(torch.tensor((np.array(rows) - xMean) / xStd)).numpy() * yStd + yMean
    with open(os.path.join(nnDir, 'nnTest.txt'), 'w') as f:
        f.write('# inputs (' + ' '.join(features) + ')  ->  outputs (' +
                ' '.join(targets) + ')\n')
        for x, y in zip(rows, out):
            f.write(' '.join(f'{a:.10g}' for a in x) + '  ->  ' +
                    ' '.join(f'{a:.10g}' for a in y) + '\n')
    print(open(os.path.join(nnDir, 'nnTest.txt')).read())
    torch.save(model.state_dict(), os.path.join(nnDir, 'model.pt'))


def geometry(runDir, outDir):
    """Source columns of a case on its simulation grid (grid 0 of samples.npz)."""
    d = np.load(os.path.join(runDir, 'samples.npz'))
    g0 = (d['grid'] == 0) & (d['volFrac'] > 0)       # no body piece -> no theta: skipped
    skipped = (d['grid'] == 0) & ~(d['volFrac'] > 0)
    e = np.column_stack([d['ePar_x'], d['ePar_y']])
    ePerp = np.column_stack([-e[:, 1], e[:, 0]])
    N = d['Npar'][:, None] * e + d['Nperp'][:, None] * ePerp
    r = (d['rpar'][:, None] * e + d['rperp'][:, None] * ePerp) * d['h'][:, None]
    cols = np.column_stack([d['colX'], d['colY'], d['h'], d['D'], d['volFrac'], N, r])[g0]
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, 'nnGeometry.txt'), 'w') as f:
        f.write(f'{newRun.gamma} {newRun.R_gas} {newRun.C_suth} {newRun.S_suth}\n')
        for c in cols:
            f.write(' '.join(f'{a:.17g}' for a in c) + '\n')
    lost = d['Sx'][skipped].sum() / d['Sx'][d['grid'] == 0].sum()
    print(f'{len(cols)} source columns -> {outDir}/nnGeometry.txt '
          f'({skipped.sum()} skipped, {100 * lost:.2f}% of the drag)')


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(*sys.argv[2:3], *[os.path.join(here, a) for a in sys.argv[3:4]])
    else:
        geometry(sys.argv[2], sys.argv[3])
