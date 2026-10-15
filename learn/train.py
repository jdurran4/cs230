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
  python -m learn.train cv [--mirror] [set ...]           compare input sets (default: all)
  python -m learn.train export [--mirror] SET NAME        -> DATA/nn/NAME/
  --mirror: add a y-mirrored copy of every TRAINING row (x-y sets only): the problem is
  symmetric under y -> -y, and the copy doubles the coverage of rare column geometries.
  --data FILE: dataset in DATA/datasets/ (default coarse_steady.npz; coarse_unsteady.npz
  from python -m learn.dataset unsteady, where Cd/Cl are scored per (run, snapshot, level)).
  Repeatable: the files' rows are combined, e.g. --data coarse_unsteady.npz --data
  coarse_steady3d.npz trains on both. CV then scores only the FIRST file's rows (the
  other files only add training rows), so the scores compare with that file alone.
  Runs are matched by case name across files (one fold; --exclude drops all of them).
  --exclude NAME: leave run NAME out entirely (repeatable), e.g. for a held-out online test

---- nnWeights.txt (plain numbers, whitespace separated, read in this order) ----------
    tag                       input set name (e.g. xy): AERO-F checks it, because
                              nets with different inputs can have the same sizes
    L                         number of linear layers (3)
    n0 n1 ... nL              layer widths (e.g. 9 32 32 2)
    xMean[n0]  xStd[n0]       input standardization:  z = (x - xMean) / xStd
    for each layer l = 1..L:
        W[n_l][n_{l-1}]       row by row (row = one output neuron)
        b[n_l]
    yMean[nL]  yStd[nL]       output de-standardization:  y = out * yStd + yMean
  Forward pass:  a = z;  a = tanh(W1 a + b1);  a = tanh(W2 a + b2);  out = W3 a + b3
  The input order and the output frame are listed in features.py and in the header of
  nnTest.txt.
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
dataFiles = []             # --data FILE, repeatable (default ['coarse_steady.npz'])
mirror = False             # set by --mirror
exclude = []               # run names left out of the data (--exclude NAME, repeatable)
torch.set_num_threads(4)


# ============= Data =================================================================
def load():
    """Rows of every --data file, concatenated, with 'dataset' = the file's index. 'run'
    is renumbered by case name over all files; 'group' (one Cd/Cl) is (file, run,
    snapshot, level), so a steady file without snapshots or levels has one per run."""
    parts, runNames = [], []
    for i, name in enumerate(dataFiles or ['coarse_steady.npz']):
        raw = np.load(os.path.join(common.DATA, 'datasets', name))
        p = {k: raw[k] for k in raw.files if k not in ('runNames', 'group')}
        n = len(p['run'])
        for k in ('snap', 'level', 't'):
            p.setdefault(k, np.zeros(n))
        names = [str(s) for s in raw['runNames']]
        runNames += [s for s in names if s not in runNames]
        p['run'] = np.array([runNames.index(s) for s in names], float)[p['run'].astype(int)]
        p['dataset'] = np.full(n, float(i))
        parts.append(p)
        print(f'{name}: {n} rows, {len(names)} runs')
    keys = set(parts[0])
    assert all(set(p) == keys for p in parts), 'the --data files have different columns'
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    d['group'] = ((d['dataset'] * 100000 + d['run']) * 10000 + d['snap']) * 10 + d['level']

    F = ft.candidates(d)
    used = sorted({f for names, _ in ft.inputSets.values() for f in names})
    keep = (d['duplicate'] == 0) & np.all([np.isfinite(F[f]) for f in used], axis=0)
    if exclude:
        keep &= ~np.isin(d['run'], [runNames.index(n) for n in exclude])
        print(f'excluded runs: {", ".join(exclude)}')
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


def augment(X, Y, names, frame):
    """Training rows, plus their y-mirror images if --mirror."""
    if not mirror:
        return X, Y
    assert frame == 'xy', 'mirroring is implemented for the x-y frame only'
    sx, sy = ft.mirrorSigns(names), np.array([1.0, -1.0])
    return np.vstack([X, X * sx]), np.vstack([Y, Y * sy])


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
        tr, te = fold != k, (fold == k) & (d['dataset'] == 0)   # score the first file only
        model, stats = fit(*augment(X[tr], Y[tr], names, frame), seed)
        teRows = {key: v[te] for key, v in d.items()}
        S = ft.toSource(predict(model, stats, X[te]), teRows, frame) / norm[te]
        sq.append((S - true[te]) ** 2)
        group = d['group'][te]                  # Cd per run (steady), per layer (unsteady)
        for r in np.unique(group):
            m = group == r
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
    first = d['dataset'] == 0
    typical = np.sqrt(np.mean((np.column_stack([d['Sx'], d['Sy']])[first] / norm[first]) ** 2))
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
    assert ' ' not in setName, 'the tag is read as one whitespace-separated token'
    d, F = load()
    names, frame = ft.inputSets[setName]
    X = np.column_stack([F[f] for f in names])
    model, (xMean, xStd, yMean, yStd) = fit(*augment(X, ft.target(d, frame), names, frame),
                                            seeds[0])
    outDir = os.path.join(common.DATA, 'nn', outName)
    os.makedirs(outDir, exist_ok=True)

    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    line = lambda v: ' '.join(f'{a:.17g}' for a in v) + '\n'
    with open(os.path.join(outDir, 'nnWeights.txt'), 'w') as f:
        f.write(f'{setName}\n')
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
        f.write(f'# input set {setName!r}, target frame {frame!r}, mirror {mirror}\n')
        f.write(f'# inputs ({" ".join(names)})  ->  outputs (Y1 Y2)\n')
        for x, y in zip(rows, out):
            f.write(' '.join(f'{a:.10g}' for a in x) + '  ->  ' +
                    ' '.join(f'{a:.10g}' for a in y) + '\n')
    torch.save(model.state_dict(), os.path.join(outDir, 'model.pt'))
    print(f'{len(X)} rows, {setName!r} -> {outDir}')


if __name__ == '__main__':
    args = sys.argv[1:]
    mirror = '--mirror' in args
    args = [a for a in args if a != '--mirror']
    while '--data' in args:
        i = args.index('--data')
        dataFiles.append(args[i + 1])
        del args[i:i + 2]
    while '--exclude' in args:
        i = args.index('--exclude')
        exclude.append(args[i + 1])
        del args[i:i + 2]
    if args[0] == 'cv':
        cv(args[1:] or list(ft.inputSets))
    else:
        export(args[1], args[2])
