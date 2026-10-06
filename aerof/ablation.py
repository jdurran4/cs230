"""
Phase 4 ablation: which inputs and which target normalization predict the source best?

Trains the SAME small MLP on every (input set, target) pair and compares test errors.
Only the inputs and the target change between fits, so differences in error come from them.

Data: dataset.npz from buildDataset.py, one row per (run, force cell), steady state, or
dataset_multi.npz from binMulti.py, one row per (run, coarse grid, force cell).
  - the _me2M cap-test runs are dropped (near copies of their 1M runs)
  - rows with a NaN in any candidate input are dropped (the 2 sliver cells), for ALL
    input sets, so every fit sees the same rows

Test split: grouped k-fold. Rows are grouped (by run, or by (M, Re), see splitBy), the
groups are shuffled and dealt into k folds, and each fold is the test set once. A run's
cells are never split between train and test (they are strongly correlated).

Score: every target is converted back to the SAME physical quantity, S / (q_inf D) in x-y,
so different normalizations compare fairly:
  cellRMSE   RMS error of S / (q_inf D) over the test cells, both components
  CdErr      RMS over test (run, grid) pairs of the error in Cd = -sum over the pair's
             cells of Sx / (q_inf D)
Each score is averaged over the k folds, then mean +- std over the seeds.

Baselines: 'mean' (predict the training mean of the target) and 'linear' (least squares
with a bias). A useful network has to beat 'linear' clearly.

Usage: set the parameters below, then  ../.venv/bin/python ablation.py [run|MRe] [dataset]
(dataset: dataset.npz (default) or dataset_multi.npz).
Writes ablation_<splitBy>.csv, or ablation_multi_<splitBy>.csv for dataset_multi.npz
(one line per input set x target x model).
"""
import os
import sys
import time

import numpy as np
import torch

# ============= User-defined parameters ==============================================
nFolds = 5
seeds = [0, 1, 2]
splitBy = sys.argv[1] if len(sys.argv) > 1 else 'run'   # 'run': unseen runs; 'MRe': unseen
                                                        # (Mach, Re) pairs (harder)
datasetFile = sys.argv[2] if len(sys.argv) > 2 else 'dataset.npz'
hidden = [32, 32]          # MLP hidden layer widths
nEpochs = 3000             # full-batch gradient steps
lr = 3e-3                  # Adam learning rate
weightDecay = 1e-4         # L2 penalty on the weights (mild regularization)

base = ['Mbar', 'logReD', 'volFrac', 'Npar', 'Nperp']
inputSets = {
    'base':             base,
    'base+theta':       base + ['cosT', 'sinT'],
    'base+dh':          base + ['dh'],
    'base+theta+dh':    base + ['cosT', 'sinT', 'dh'],
    'base+r':           base + ['rpar', 'rperp'],
    'base+theta+dh+Dh': base + ['cosT', 'sinT', 'dh', 'Dh'],
    'all':              base + ['cosT', 'sinT', 'dh', 'rpar', 'rperp', 'Nmag', 'Dh'],
}
# leave-one-out from the richest physical set: what is each input worth?
# (Dh is constant in dataset.npz, so those fits only matter for dataset_multi.npz)
for name in inputSets['base+theta+dh+Dh']:
    inputSets[f'b+t+dh+Dh -{name}'] = [f for f in inputSets['base+theta+dh+Dh'] if f != name]

# target name -> the two dataset columns predicted (x, y components or par, perp)
targets = {
    'SInf':   ('SxInf', 'SyInf'),          # S / (q_inf D), x-y frame
    'Sp_q':   ('Spar_q', 'Sperp_q'),       # Sp . e / (qbar D)
    'Sp_pq':  ('Spar_pq', 'Sperp_pq'),     # Sp . e / ((pbar + qbar) D)
    'SF_q':   ('SFpar_q', 'SFperp_q'),     # S . e / (qbar D)
    'SF_pq':  ('SFpar_pq', 'SFperp_pq'),   # S . e / ((pbar + qbar) D)
}

aerofDir = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)   # tiny problem: threading overhead costs more than it saves


# ============= Data =================================================================
def loadData():
    """dataset.npz -> dict of 1-D numpy arrays (one entry per row), filtered."""
    raw = np.load(os.path.join(aerofDir, datasetFile))
    d = {k: raw[k] for k in raw['featureNames']}
    d.setdefault('grid', np.zeros_like(d['run']))  # single-grid dataset: grid 0 only
    d['logReD'] = np.log(d['ReD'])             # Re_D spans 0.7 .. 170: log is better scaled
    candidates = sorted({f for s in inputSets.values() for f in s})
    keep = (d['duplicate'] == 0) & np.all([np.isfinite(d[f]) for f in candidates], axis=0)
    d = {k: v[keep] for k, v in d.items()}
    print(f'{keep.sum()} rows kept of {len(keep)}, {len(np.unique(d["run"]))} runs')
    return d


def groupFolds(d, seed):
    """Fold index (0 .. nFolds-1) of every row, dealing shuffled groups into folds."""
    if splitBy == 'run':
        group = d['run']
    else:
        group = d['machInf'] * 1000 + d['ReInf']
    groups = np.unique(group)
    rng = np.random.default_rng(seed)
    foldOfGroup = dict(zip(rng.permutation(groups), np.arange(len(groups)) % nFolds))
    return np.array([foldOfGroup[g] for g in group])


def toSInf(targetName, pred, d, rows):
    """Convert predictions (n, 2) of a target back to S / (q_inf D) in x-y (n, 2)."""
    e = np.column_stack([d['ePar_x'][rows], d['ePar_y'][rows]])           # flow direction
    ePerp = np.column_stack([-e[:, 1], e[:, 0]])
    D, qInf = d['D'][rows], d['qInf'][rows]
    pBar, qBar = d['pBar'][rows], d['qBar'][rows]
    if targetName == 'SInf':
        return pred
    scale = qBar * D if targetName.endswith('_q') else (pBar + qBar) * D
    vec = scale[:, None] * (pred[:, :1] * e + pred[:, 1:] * ePerp)       # back to x-y, N/m
    if targetName.startswith('Sp'):                                       # add pbar D N back
        N = d['Npar'][rows, None] * e + d['Nperp'][rows, None] * ePerp
        vec = vec + (pBar * D)[:, None] * N
    return vec / (qInf * D)[:, None]


# ============= Models ===============================================================
def makeMLP(nIn):
    """Fully connected network: nIn -> hidden ... -> 2, tanh between layers.
    nn.Sequential just applies its modules in order; nn.Linear(a, b) is y = W x + c with
    W (b, a) and c (b,) as trainable parameters."""
    layers, width = [], nIn
    for h in hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.Tanh()]
        width = h
    layers.append(torch.nn.Linear(width, 2))
    return torch.nn.Sequential(*layers)


def fitMLP(Xtr, Ytr, Xte, seed):
    """Train on (Xtr, Ytr), return predictions on Xte. All arrays standardized numpy."""
    torch.manual_seed(seed)                     # same initial weights for the same seed
    model = makeMLP(Xtr.shape[1])
    # Tensors are torch's arrays; float32 is the usual training precision
    X = torch.tensor(Xtr, dtype=torch.float32)
    Y = torch.tensor(Ytr, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
    for epoch in range(nEpochs):
        opt.zero_grad()                         # clear gradients from the previous step
        loss = torch.mean((model(X) - Y) ** 2)  # forward pass: MSE on the training set
        loss.backward()                         # backprop: d loss / d every parameter
        opt.step()                              # Adam update of every parameter
    with torch.no_grad():                       # inference only: no gradient bookkeeping
        return model(torch.tensor(Xte, dtype=torch.float32)).numpy()


def fitLinear(Xtr, Ytr, Xte, seed):
    A = np.column_stack([Xtr, np.ones(len(Xtr))])
    W = np.linalg.lstsq(A, Ytr, rcond=None)[0]
    return np.column_stack([Xte, np.ones(len(Xte))]) @ W


def fitMean(Xtr, Ytr, Xte, seed):
    return np.tile(Ytr.mean(axis=0), (len(Xte), 1))


models = {'mlp': fitMLP, 'linear': fitLinear, 'mean': fitMean}


# ============= One cross-validated score ============================================
def crossValidate(d, features, targetName, fit, seed):
    """Mean over folds of (cellRMSE, CdErr) for one input set, target, model, seed."""
    X = np.column_stack([d[f] for f in features])
    Y = np.column_stack([d[c] for c in targets[targetName]])
    trueSInf = np.column_stack([d['SxInf'], d['SyInf']])
    fold = groupFolds(d, seed)
    cellErr, cdErr = [], []
    for k in range(nFolds):
        tr, te = fold != k, fold == k
        # standardize with TRAINING statistics only (the test set stays unseen)
        xm, xs = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-12
        ym, ys = Y[tr].mean(axis=0), Y[tr].std(axis=0) + 1e-12
        pred = fit((X[tr] - xm) / xs, (Y[tr] - ym) / ys, (X[te] - xm) / xs, seed) * ys + ym
        S = toSInf(targetName, pred, d, np.flatnonzero(te))
        cellErr.append(np.sqrt(np.mean((S - trueSInf[te]) ** 2)))
        runs = d['run'][te] * 1000 + d['grid'][te]                      # (run, grid) pairs
        dCd = [S[runs == r, 0].sum() - trueSInf[te][runs == r, 0].sum() for r in np.unique(runs)]
        cdErr.append(np.sqrt(np.mean(np.square(dCd))))
    return np.mean(cellErr), np.mean(cdErr)


# ============= Main =================================================================
if __name__ == '__main__':
    d = loadData()
    # sanity check: converting the TRUE targets back must reproduce S / (q_inf D)
    allRows = np.arange(len(d['run']))
    for t, cols in targets.items():
        back = toSInf(t, np.column_stack([d[c] for c in cols]), d, allRows)
        err = np.abs(back - np.column_stack([d['SxInf'], d['SyInf']])).max()
        assert err < 1e-6, f'{t}: round trip error {err:.1e}'

    results = []
    t0 = time.time()
    for setName, features in inputSets.items():
        for targetName in targets:
            for modelName, fit in models.items():
                if modelName == 'mean' and setName != 'base':
                    continue                    # the mean ignores the inputs
                scores = np.array([crossValidate(d, features, targetName, fit, s)
                                   for s in seeds])                         # (nSeeds, 2)
                m, s = scores.mean(axis=0), scores.std(axis=0)
                results.append((setName, targetName, modelName, m[0], s[0], m[1], s[1]))
                print(f'{time.time() - t0:6.0f}s  {setName:<18} {targetName:<6} '
                      f'{modelName:<6}  cellRMSE {m[0]:.4f} +- {s[0]:.4f}   '
                      f'CdErr {m[1]:.4f} +- {s[1]:.4f}', flush=True)

    results.sort(key=lambda r: r[3])
    print(f'\nsorted by cellRMSE (S / (q_inf D)), split by {splitBy}, {nFolds} folds, '
          f'{len(seeds)} seeds; typical |S / (q_inf D)| of a cell: '
          f'{np.sqrt(np.mean(d["SxInf"] ** 2 + d["SyInf"] ** 2) / 2):.4f} RMS')
    print(f'{"input set":<18} {"target":<6} {"model":<6} {"cellRMSE":>17} {"CdErr":>17}')
    for r in results:
        print(f'{r[0]:<18} {r[1]:<6} {r[2]:<6} {r[3]:9.4f} +- {r[4]:.4f} '
              f'{r[5]:9.4f} +- {r[6]:.4f}')
    tag = 'multi_' if 'multi' in datasetFile else ''
    out = os.path.join(aerofDir, f'ablation_{tag}{splitBy}.csv')
    with open(out, 'w') as f:
        f.write('inputSet,target,model,cellRMSE,cellRMSEstd,CdErr,CdErrStd\n')
        for r in results:
            f.write(','.join(map(str, r[:3])) + ',' + ','.join(f'{v:.6g}' for v in r[3:]) + '\n')
    print(f'wrote {out}')
