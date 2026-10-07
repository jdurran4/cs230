"""A priori test of the distribution-shift fix: fine-trained vs coarse-trained network,
both evaluated on what the network sees online (coarse oracle states, held-out runs).

5 folds split by run. Per fold:
  fine    trained on dataset_multi.npz (fine averages, all 21 grids) of the training runs
  coarse  trained on dataset_coarse.npz (coarse oracle states, grid 0) of the training runs
  both tested on dataset_coarse.npz rows of the held-out runs. Error in S / (q_inf D) per
cell, and the Cd error per held-out run (sum over its cells), as in ablation.py.
Same network and training settings as nnExport.py.

Usage:  python nnCoarseCompare.py
"""
import numpy as np
import torch

import nnExport as nx

torch.set_num_threads(8)


def load(name):
    raw = np.load(f'{nx.here}/{name}')
    d = {k: raw[k] for k in raw['featureNames']}
    d['logReD'] = np.log(d['ReD'])
    keep = (d['duplicate'] == 0) & np.all([np.isfinite(d[f]) for f in nx.features + nx.targets],
                                          axis=0)
    return {k: v[keep] for k, v in d.items()}


def fit(d, rows, seed):
    X = np.column_stack([d[f][rows] for f in nx.features])
    Y = np.column_stack([d[t][rows] for t in nx.targets])
    xm, xs, ym, ys = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
    xs[xs < 1e-8 * np.maximum(np.abs(xm), 1.0)] = 1.0      # constant input (Dh, grid 0)
    torch.manual_seed(seed)
    layers, w = [], len(nx.features)
    for h in nx.hidden:
        layers += [torch.nn.Linear(w, h), torch.nn.Tanh()]
        w = h
    layers.append(torch.nn.Linear(w, 2))
    model = torch.nn.Sequential(*layers).double()
    Xt, Yt = torch.tensor((X - xm) / xs), torch.tensor((Y - ym) / ys)
    opt = torch.optim.Adam(model.parameters(), lr=nx.lr, weight_decay=nx.weightDecay)
    for _ in range(nx.nEpochs):
        opt.zero_grad()
        loss = torch.mean((model(Xt) - Yt) ** 2)
        loss.backward()
        opt.step()

    def predict(dd, r):
        Xe = np.column_stack([dd[f][r] for f in nx.features])
        with torch.no_grad():
            return model(torch.tensor((Xe - xm) / xs)).numpy() * ys + ym
    return predict


def toSInf(pred, d, r):
    """Sp_pq prediction -> S / (q_inf D) in x-y, with the row's own pbar, qbar, frame."""
    e = np.column_stack([d['ePar_x'][r], d['ePar_y'][r]])
    ep = np.column_stack([-e[:, 1], e[:, 0]])
    D, pB, qB = d['D'][r], d['pBar'][r], d['qBar'][r]
    S = ((pB + qB) * D)[:, None] * (pred[:, :1] * e + pred[:, 1:] * ep)
    N = d['Npar'][r, None] * e + d['Nperp'][r, None] * ep
    S = S + (pB * D)[:, None] * N
    return S / (d['qInf'][r] * D)[:, None]


fine, coarse = load('dataset_multi.npz'), load('dataset_coarse.npz')
runs = np.unique(coarse['run'])
rng = np.random.default_rng(0)
fold = dict(zip(rng.permutation(runs), np.arange(len(runs)) % 5))
res = {k: dict(cell=[], cd=[], cl=[]) for k in ('fine', 'coarse')}
for f in range(5):
    test = np.array([fold[r] == f for r in coarse['run']])
    trainRuns = {r for r in runs if fold[r] != f}
    rows = {'fine': np.flatnonzero([r in trainRuns for r in fine['run']]),
            'coarse': np.flatnonzero(~test)}
    te = np.flatnonzero(test)
    true = np.column_stack([coarse['SxInf'][te], coarse['SyInf'][te]])
    for name, d in (('fine', fine), ('coarse', coarse)):
        S = toSInf(fit(d, rows[name], seed=f)(coarse, te), coarse, te)
        res[name]['cell'].append(np.sqrt(np.mean((S - true) ** 2)))
        for r in np.unique(coarse['run'][te]):
            m = coarse['run'][te] == r
            res[name]['cd'].append(S[m, 0].sum() - true[m, 0].sum())
            res[name]['cl'].append(S[m, 1].sum() - true[m, 1].sum())
    print(f'fold {f}: cell RMSE fine {res["fine"]["cell"][-1]:.4f}, '
          f'coarse {res["coarse"]["cell"][-1]:.4f}')

rms = lambda a: np.sqrt(np.mean(np.square(a)))
print(f'\ntested on coarse oracle states of held-out runs ({len(runs)} runs, 5 folds)')
print(f'{"trained on":12s} {"cell RMSE":>10s} {"Cd err RMS":>11s} {"Cd err mean":>12s} '
      f'{"Cl err RMS":>11s}')
for name, r in res.items():
    print(f'{name:12s} {np.mean(r["cell"]):10.4f} {rms(r["cd"]):11.4f} {np.mean(r["cd"]):+12.4f} '
          f'{rms(r["cl"]):11.4f}')
print('(Cd err = predicted minus true drag coefficient per run; true Cd is about 1.5-1.9)')
