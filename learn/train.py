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
  train  cell RMSE on the rows each fold trained on (not the mirror copies), pooled over
         the folds: train far below cell = overfitting, train close to cell = underfitting
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
  coarse_steady3d.npz trains on both. The main CV line scores only the FIRST file's rows,
  so it compares with that file alone; with several files a line per file follows (one
  file is one D/h), each scored on that file's held-out rows.
  Runs are matched by case name across files (one fold; --exclude drops all of them).
  --exclude NAME: leave run NAME out entirely (repeatable), e.g. for a held-out online test
  --weightDh A: weight each row's loss by (D/h)^A, to balance meshes in a shared net
  --holdDh X: leave-one-mesh-out: rows with D/h = X are never trained on (the folds are
  unchanged, so X's line is scored on unseen runs of an unseen mesh)
  Environment: HIDDEN=64,64 (hidden widths), SEEDS=0 (CV seeds, e.g. for a quick screen)

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
# HIDDEN=64,64 and SEEDS=0 (environment) override the widths and the CV seeds.
hidden = [int(n) for n in os.environ.get('HIDDEN', '32,32').split(',')]   # hidden widths
nEpochs = 3000             # full-batch Adam steps
lr = 3e-3                  # learning rate
weightDecay = 1e-4         # L2 penalty on the weights
nFolds = 5
seeds = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
weightDh = 0.0             # --weightDh A: loss weight of a row ~ (D/h)^A (0 = unweighted).
                           # The target scales ~ h/D, so A = 2 weights every row's relative
                           # error equally; A = 1 gives each mesh about the same share of the
                           # loss (the finer mesh has ~D/h times more rows).
dataFiles = []             # --data FILE, repeatable (default ['coarse_steady.npz'])
mirror = False             # set by --mirror
exclude = []               # run names left out of the data (--exclude NAME, repeatable)
holdDh = None              # --holdDh X: never train on the rows with D/h = X
device = 'cuda' if torch.cuda.is_available() else 'cpu'    # fit only; predict on the CPU
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
        for k in ('snap', 'level', 't', 'pRef'):       # older files: absolute pressure
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
    assert np.all(d['pRef'] == 0) or np.all(d['pRef'] > 0), \
        'the --data files mix absolute and p - p_inf sources'
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
def fit(X, Y, seed, W=None):
    """Train an MLP on (X, Y) (numpy, one row per sample) and return it with the
    standardization it was trained with. float64 throughout: the C++ uses double.
    W: optional weight per row in the loss (rescaled to mean 1)."""
    xMean, xStd = X.mean(axis=0), X.std(axis=0)
    xStd[xStd < 1e-8 * np.maximum(np.abs(xMean), 1.0)] = 1.0    # constant input (Dh)
    yMean, yStd = Y.mean(axis=0), Y.std(axis=0)

    torch.manual_seed(seed)                     # same initial weights for the same seed
    layers, width = [], X.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.Tanh()]
        width = h
    layers.append(torch.nn.Linear(width, Y.shape[1]))
    model = torch.nn.Sequential(*layers).double().to(device)

    Xt = torch.tensor((X - xMean) / xStd, device=device)
    Yt = torch.tensor((Y - yMean) / yStd, device=device)
    W = np.ones(len(X)) if W is None else W / W.mean()
    Wt = torch.tensor(W[:, None], device=device)    # (n, 1): same weight for both outputs
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
    for _ in range(nEpochs):
        opt.zero_grad()                         # clear the previous step's gradients
        loss = torch.mean(Wt * (model(Xt) - Yt) ** 2)
        loss.backward()                         # d loss / d every weight
        opt.step()
    return model.cpu(), (xMean, xStd, yMean, yStd)


def augment(X, Y, W, names, frame):
    """Training rows and their loss weights, plus the rows' y-mirror images if --mirror."""
    if not mirror:
        return X, Y, W
    assert frame == 'xy', 'mirroring is implemented for the x-y frame only'
    sx, sy = ft.mirrorSigns(names), np.array([1.0, -1.0])
    return np.vstack([X, X * sx]), np.vstack([Y, Y * sy]), np.concatenate([W, W])


def rowWeights(d):
    return (d['D'] / d['h']) ** weightDh


def predict(model, stats, X):
    xMean, xStd, yMean, yStd = stats
    with torch.no_grad():
        return model(torch.tensor((X - xMean) / xStd)).numpy() * yStd + yMean


# ============= Cross-validation =====================================================
def crossValidate(d, F, setName, seed):
    """Scores (nFiles, 5) per --data file: cell RMSE, Cd and Cl error RMS, Cd bias, each
    on that file's rows while they were held out, and the train cell RMSE."""
    names, frame = ft.inputSets[setName]
    X = np.column_stack([F[f] for f in names])
    Y = ft.target(d, frame)
    norm = (d['qInf'] * d['D'])[:, None]                         # S -> S / (q_inf D)
    true = np.column_stack([d['Sx'], d['Sy']]) / norm
    runs = np.unique(d['run'])
    rng = np.random.default_rng(seed)
    foldOf = dict(zip(rng.permutation(runs), np.arange(len(runs)) % nFolds))
    fold = np.array([foldOf[r] for r in d['run']])

    trainable = np.ones(len(X), bool)
    if holdDh is not None:
        trainable = ~np.isclose(d['D'] / d['h'], holdDh, rtol=1e-3)
        assert 0 < trainable.sum() < len(X), f'--holdDh {holdDh}: no such mesh'
    err = np.empty_like(true)                   # held-out prediction - true, every row
    trainSq = np.zeros(len(X))                  # squared train error summed over folds
    trainN = np.zeros(len(X))                   # how many folds trained on the row
    for k in range(nFolds):
        tr, te = (fold != k) & trainable, fold == k
        Xa, Ya, Wa = augment(X[tr], Y[tr], rowWeights(d)[tr], names, frame)
        model, stats = fit(Xa, Ya, seed, Wa)
        teRows = {key: v[te] for key, v in d.items()}
        err[te] = ft.toSource(predict(model, stats, X[te]), teRows, frame) / norm[te] - true[te]
        trRows = {key: v[tr] for key, v in d.items()}
        e = ft.toSource(predict(model, stats, X[tr]), trRows, frame) / norm[tr] - true[tr]
        trainSq[tr] += np.sum(e ** 2, axis=1)
        trainN[tr] += 1

    rms = lambda a: np.sqrt(np.mean(np.square(a)))
    scores = []
    for i in np.unique(d['dataset']):
        m = d['dataset'] == i
        # Cd per run (steady), per layer (unsteady)
        _, group = np.unique(d['group'][m], return_inverse=True)
        dCd, dCl = -np.bincount(group, err[m, 0]), -np.bincount(group, err[m, 1])
        n = trainN[m].sum()                     # nan for a held-out mesh
        trainRMSE = np.sqrt(trainSq[m].sum() / (2 * n)) if n else np.nan
        scores.append((rms(err[m]), rms(dCd), rms(dCl), np.mean(dCd), trainRMSE))
    return np.array(scores)


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
          f'{"Cd bias":>8} {"train":>7}')
    t0 = time.time()
    for name in setNames:
        s = np.array([crossValidate(d, F, name, seed) for seed in seeds])  # (seeds, files, 4)
        m, sd = s.mean(axis=0), s.std(axis=0)
        line = lambda i: (f'{m[i, 0]:8.4f} +- {sd[i, 0]:.4f} {m[i, 1]:8.4f} +- {sd[i, 1]:.4f} '
                          f'{m[i, 2]:8.4f} +- {sd[i, 2]:.4f} {m[i, 3]:+8.4f} {m[i, 4]:7.4f}')
        print(f'{name:<16} {line(0)}   ({time.time() - t0:.0f} s)', flush=True)
        for i, file in enumerate(dataFiles if len(dataFiles) > 1 else []):
            Dh = (d['D'] / d['h'])[d['dataset'] == i][0]
            print(f'  D/h {Dh:<4g}{line(i)}   {file}', flush=True)


# ============= Export ===============================================================
def export(setName, outName):
    assert ' ' not in setName, 'the tag is read as one whitespace-separated token'
    d, F = load()
    names, frame = ft.inputSets[setName]
    X = np.column_stack([F[f] for f in names])
    Xa, Ya, Wa = augment(X, ft.target(d, frame), rowWeights(d), names, frame)
    model, (xMean, xStd, yMean, yStd) = fit(Xa, Ya, seeds[0], Wa)
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
    if '--holdDh' in args:
        i = args.index('--holdDh')
        holdDh = float(args[i + 1])
        del args[i:i + 2]
    if '--weightDh' in args:
        i = args.index('--weightDh')
        weightDh = float(args[i + 1])
        del args[i:i + 2]
    print(f'hidden {hidden}, seeds {seeds}, weightDh {weightDh:g}, mirror {mirror}'
          + (f', holdDh {holdDh:g}' if holdDh is not None else ''))
    if args[0] == 'cv':
        cv(args[1:] or list(ft.inputSets))
    else:
        export(args[1], args[2])
