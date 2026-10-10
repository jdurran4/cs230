"""Train the source-term MLP: cross-validated comparison of input sets, and export.

Data: DATA/datasets/coarse_steady.npz (learn/dataset.py), inputs and targets from
learn/features.py. Rows: the _me2M / zo0.1 duplicate runs are dropped, and rows with a
NaN in any candidate input (the sliver columns with no body volume) are dropped for
every input set, so all sets see the same rows.

Cross-validation: folds split by run (a run's columns are never split between train and
test), 5 folds, repeated for 3 seeds (new fold shuffle and initial weights). Scores, all
in the same physical units whatever the frame:
  cell   RMS error of S / (q_inf D) over the test columns, both components
  Cd, Cl RMS over test runs of the error in Cd = -sum Sx / (q_inf D) (and Cl with Sy)
reported as mean +- std over the seeds.

Export: train on all rows, write nnWeights.txt (the file AERO-F reads; format below)
and nnTest.txt (a few input rows and outputs, to check the C++ forward pass).

Usage (from the repo root):
  python -m learn.train cv [set ...]             compare input sets (default: all)
  python -m learn.train export SET NAME          -> DATA/nn/NAME/

---- nnWeights.txt (plain numbers, whitespace separated, read in this order) ----------
    L                         number of linear layers (3)
    n0 n1 ... nL              layer widths (e.g. 9 32 32 2)
    xMean[n0]  xStd[n0]       input standardization:  z = (x - xMean) / xStd
    for each layer l = 1..L:
        W[n_l][n_{l-1}]       row by row (row = one output neuron)
        b[n_l]
    yMean[nL]  yStd[nL]       output de-standardization:  y = out * yStd + yMean
  Forward pass:  a = z;  a = tanh(W1 a + b1);  a = tanh(W2 a + b2);  out = W3 a + b3
  The input order and the output frame are listed in the header of nnTest.txt.
"""
import os
import sys
import time

import numpy as np
import torch

import common
from learn import features as ft

# ============= Settings =============================================================
hidden = [32, 32]          # hidden layer widths
nEpochs = 3000             # full-batch Adam steps
lr = 3e-3                  # learning rate
weightDecay = 1e-4         # L2 penalty on the weights
nFolds = 5
seeds = [0, 1, 2]
dataFile = os.path.join(common.DATA, 'datasets', 'coarse_steady.npz')
torch.set_num_threads(4)


# ============= Data =================================================================
def load():
    raw = np.load(dataFile)
    d = {k: raw[k] for k in raw.files if k != 'runNames'}
    F = ft.candidates(d)
    used = sorted({f for names, _ in ft.inputSets.values() for f in names})
    keep = (d['duplicate'] == 0) & np.all([np.isfinite(F[f]) for f in used], axis=0)
    print(f'{keep.sum()} of {len(keep)} rows kept, {len(np.unique(d["run"][keep]))} runs')
    return ({k: v[keep] for k, v in d.items()}, {k: v[keep] for k, v in F.items()})


# ============= Model ================================================================
def fit(X, Y, seed):
    """Train an MLP on (X, Y) (numpy, one row per sample) and return it with the
    standardization it was trained with. float64 throughout: the C++ uses double."""
    xMean, xStd = X.mean(axis=0), X.std(axis=0)
    xStd[xStd < 1e-8 * np.maximum(np.abs(xMean), 1.0)] = 1.0    # constant input (Dh)
    yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

    torch.manual_seed(seed)                     # same initial weights for the same seed
    layers, width = [], X.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.Tanh()]
        width = h
    layers.append(torch.nn.Linear(width, Y.shape[1]))
    model = torch.nn.Sequential(*layers).double()

    Xt = torch.tensor((X - xMean) / xStd)
    Yt = torch.tensor((Y - yMean) / yStd)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
    for _ in range(nEpochs):
        opt.zero_grad()                         # clear the previous step's gradients
        loss = torch.mean((model(Xt) - Yt) ** 2)
        loss.backward()                         # d loss / d every weight
        opt.step()
    return model, (xMean, xStd, yMean, yStd)


def predict(model, stats, X):
    xMean, xStd, yMean, yStd = stats
    with torch.no_grad():
        return model(torch.tensor((X - xMean) / xStd)).numpy() * yStd + yMean


# ============= Cross-validation =====================================================
def crossValidate(d, F, setName, seed):
    names, frame = ft.inputSets[setName]
    X = np.column_stack([F[f] for f in names])
    Y = ft.target(d, frame)
    norm = (d['qInf'] * d['D'])[:, None]                         # S -> S / (q_inf D)
    true = np.column_stack([d['Sx'], d['Sy']]) / norm
    runs = np.unique(d['run'])
    rng = np.random.default_rng(seed)
    foldOf = dict(zip(rng.permutation(runs), np.arange(len(runs)) % nFolds))
    fold = np.array([foldOf[r] for r in d['run']])

    sq, dCd, dCl = [], [], []
    for k in range(nFolds):
        tr, te = fold != k, fold == k
        model, stats = fit(X[tr], Y[tr], seed)
        teRows = {key: v[te] for key, v in d.items()}
        S = ft.toSource(predict(model, stats, X[te]), teRows, frame) / norm[te]
        sq.append((S - true[te]) ** 2)
        for r in np.unique(d['run'][te]):
            m = d['run'][te] == r
            dCd.append(-(S[m, 0].sum() - true[te][m, 0].sum()))
            dCl.append(-(S[m, 1].sum() - true[te][m, 1].sum()))
    rms = lambda a: np.sqrt(np.mean(np.square(a)))
    return np.sqrt(np.mean(np.concatenate(sq))), rms(dCd), rms(dCl), np.mean(dCd)


def cv(setNames):
    d, F = load()
    norm = (d['qInf'] * d['D'])[:, None]
    for frame in ('flow', 'xy'):                # round trip: target -> source is exact
        back = ft.toSource(ft.target(d, frame), d, frame)
        assert np.abs(back - np.column_stack([d['Sx'], d['Sy']])).max() / norm.max() < 1e-12
    typical = np.sqrt(np.mean((np.column_stack([d['Sx'], d['Sy']]) / norm) ** 2))
    print(f'typical |S / (q_inf D)| of a column: {typical:.3f} RMS\n')
    print(f'{"input set":<16} {"cell RMSE":>17} {"Cd err RMS":>17} {"Cl err RMS":>17} '
          f'{"Cd bias":>8}')
    t0 = time.time()
    for name in setNames:
        s = np.array([crossValidate(d, F, name, seed) for seed in seeds])
        m, sd = s.mean(axis=0), s.std(axis=0)
        print(f'{name:<16} {m[0]:8.4f} +- {sd[0]:.4f} {m[1]:8.4f} +- {sd[1]:.4f} '
              f'{m[2]:8.4f} +- {sd[2]:.4f} {m[3]:+8.4f}   ({time.time() - t0:.0f} s)',
              flush=True)


# ============= Export ===============================================================
def export(setName, outName):
    d, F = load()
    names, frame = ft.inputSets[setName]
    X = np.column_stack([F[f] for f in names])
    model, (xMean, xStd, yMean, yStd) = fit(X, ft.target(d, frame), seeds[0])
    outDir = os.path.join(common.DATA, 'nn', outName)
    os.makedirs(outDir, exist_ok=True)

    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    line = lambda v: ' '.join(f'{a:.17g}' for a in v) + '\n'
    with open(os.path.join(outDir, 'nnWeights.txt'), 'w') as f:
        f.write(f'{len(linears)}\n')
        f.write(' '.join(str(n) for n in [len(names)] + hidden + [2]) + '\n')
        f.write(line(xMean) + line(xStd))
        for lin in linears:
            for row in lin.weight.detach().numpy():
                f.write(line(row))
            f.write(line(lin.bias.detach().numpy()))
        f.write(line(yMean) + line(yStd))

    rows = X[[0, len(X) // 2, -1]]
    out = predict(model, (xMean, xStd, yMean, yStd), rows)
    with open(os.path.join(outDir, 'nnTest.txt'), 'w') as f:
        f.write(f'# input set {setName!r}, target frame {frame!r}\n')
        f.write(f'# inputs ({" ".join(names)})  ->  outputs (Y1 Y2)\n')
        for x, y in zip(rows, out):
            f.write(' '.join(f'{a:.10g}' for a in x) + '  ->  ' +
                    ' '.join(f'{a:.10g}' for a in y) + '\n')
    torch.save(model.state_dict(), os.path.join(outDir, 'model.pt'))
    print(f'{len(X)} rows, {setName!r} -> {outDir}')


if __name__ == '__main__':
    if sys.argv[1] == 'cv':
        cv(sys.argv[2:] or list(ft.inputSets))
    else:
        export(sys.argv[2], sys.argv[3])
