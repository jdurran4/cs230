"""Slide figures and numbers for the network's a priori accuracy (on the coarse oracle
states, before any online run): the combined (unsteady + steady3d) x-y network with
--mirror, the one the online M1.5 showcase uses.

  parity.pdf   predicted vs oracle Sx, Sy per column (S / (q_inf D)). Grey: every unsteady
               row, predicted by the net of the CV fold that held its case out (5 folds
               by case, seed 0), so no point was seen in training. Colour: the M1.5 case,
               predicted by nn/xy_comb_noM15 (the exported net of the online test).
  apriori.pdf  Cd(t), Cl(t) of M1.5: the oracle source vs xy_comb_noM15 evaluated on the
               oracle's coarse states (mean over the interior z-levels).
  accuracy.txt the numbers: training fit (all rows, seed 0) vs held-out (CV) errors.

Data: datasets/coarse_unsteady_8runs.npz (the 8 unsteady cases the showcase nets were
trained on; frozen before the batch-2 rebuild of coarse_unsteady.npz) + coarse_steady3d.npz.
Usage (from the repo root):  python -m learn.slideAccuracy
Writes DATA/unsteady/online/slides/{parity,apriori}.{pdf,png}, accuracy.txt.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common
from learn import features as ft
from learn import train
from online import unsteady
from sweep import newRun, oracle

# ============= User-defined parameters ==============================================
dataFiles = ['coarse_unsteady_8runs.npz', 'coarse_steady3d.npz']
setName = 'xy'
showcase = 'M1.5_Re100_x0.04_y0.12'
showcaseNet = 'xy_comb_noM15'
seed = 0
outDir = os.path.join(common.UNSTEADY, 'online', 'slides')
grey = '#9a9893'


# ============= Network from nnWeights.txt ============================================
def loadNet(netName):
    """Forward pass of DATA/nn/<netName>/nnWeights.txt (format in learn/train.py)."""
    with open(os.path.join(common.DATA, 'nn', netName, 'nnWeights.txt')) as fh:
        lines = fh.read().splitlines()
    assert lines[0] == setName
    nLayers = int(lines[1])
    widths = [int(a) for a in lines[2].split()]
    vec = iter(np.array(l.split(), float) for l in lines[3:])
    xMean, xStd = next(vec), next(vec)
    layers = []
    for l in range(nLayers):
        W = np.array([next(vec) for _ in range(widths[l + 1])])
        layers.append((W, next(vec)))
    yMean, yStd = next(vec), next(vec)

    def forward(X):
        a = (X - xMean) / xStd
        for l, (W, b) in enumerate(layers):
            a = a @ W.T + b
            if l < nLayers - 1:
                a = np.tanh(a)
        return a * yStd + yMean
    return forward


# ============= Predictions ==========================================================
def heldOutPredictions(d, X, Y, names, frame):
    """S / (q_inf D) of every first-file row from the CV fold that held its case out
    (same fold split as train.crossValidate for this seed)."""
    norm = (d['qInf'] * d['D'])[:, None]
    runs = np.unique(d['run'])
    rng = np.random.default_rng(seed)
    foldOf = dict(zip(rng.permutation(runs), np.arange(len(runs)) % train.nFolds))
    fold = np.array([foldOf[r] for r in d['run']])
    S = np.full((len(fold), 2), np.nan)
    for k in range(train.nFolds):
        tr, te = fold != k, (fold == k) & (d['dataset'] == 0)
        if not te.any():
            continue
        model, stats = train.fit(*train.augment(X[tr], Y[tr], names, frame), seed)
        rows = {key: v[te] for key, v in d.items()}
        S[te] = ft.toSource(train.predict(model, stats, X[te]), rows, frame) / norm[te]
        print(f'fold {k}: {te.sum()} unsteady test rows', flush=True)
    return S


def trainingFit(d, X, Y, names, frame):
    """S / (q_inf D) of every row from one net trained on all rows (training error)."""
    model, stats = train.fit(*train.augment(X, Y, names, frame), seed)
    norm = (d['qInf'] * d['D'])[:, None]
    return ft.toSource(train.predict(model, stats, X), d, frame) / norm


def coefficients(d, S):
    """Cd, Cl per group (= -sum over the layer's columns of S / (q_inf D))."""
    groups, inv = np.unique(d['group'], return_inverse=True)
    C = np.zeros((len(groups), 2))
    np.add.at(C, inv, -S)
    return groups, C


# ============= Figures ==============================================================
def tidy(ax):
    ax.grid(color='#e1e0d9', lw=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def save(fig, stem):
    os.makedirs(outDir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outDir, f'{stem}.{ext}'), dpi=200, bbox_inches='tight')
    print('wrote', os.path.join(outDir, stem + '.{pdf,png}'))


def parity(true, cv, mShow, show):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for c, (ax, label) in enumerate(zip(axes, ('$S_x$', '$S_y$'))):
        lo = min(true[:, c].min(), cv[:, c].min())
        hi = max(true[:, c].max(), cv[:, c].max())
        ax.plot([lo, hi], [lo, hi], color='#52514e', lw=0.8, zorder=0)
        ax.scatter(true[:, c], cv[:, c], s=3, color=grey, alpha=0.35, lw=0,
                   label='all unsteady cases (held out by case)')
        ax.scatter(true[mShow, c], show[:, c], s=4, color=unsteady.colors[2], alpha=0.6,
                   lw=0, label='M1.5 showcase case (held out)')
        r2 = 1 - np.sum((cv[:, c] - true[:, c]) ** 2) / np.sum((true[:, c] - true[:, c].mean()) ** 2)
        ax.text(0.04, 0.94, f'$R^2$ = {r2:.3f}', transform=ax.transAxes, va='top')
        ax.set_xlabel(f'oracle {label} / $(q_\\infty D)$')
        ax.set_ylabel(f'NN {label} / $(q_\\infty D)$')
        ax.set_aspect('equal')
        tidy(ax)
    axes[0].legend(loc='lower right', fontsize=8, markerscale=4, frameon=False)
    fig.tight_layout()
    save(fig, 'parity')


def apriori(t, Ctrue, Cnn):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for c, (ax, label) in enumerate(zip(axes, ('$C_d$', '$C_l$'))):
        ax.plot(t, Ctrue[:, c], color=unsteady.colors[0], lw=1.8, ls='--', label='oracle')
        ax.plot(t, Cnn[:, c], color=unsteady.colors[2], lw=1.8,
                label='NN on oracle states (M1.5 held out)')
        ax.set_xlabel('$t\\,U/D$')
        ax.set_ylabel(label)
        tidy(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'apriori')


# ============= Main =================================================================
if __name__ == '__main__':
    train.dataFiles, train.mirror = dataFiles, True
    d, F = train.load()
    names, frame = ft.inputSets[setName]
    X = np.column_stack([F[f] for f in names])
    Y = ft.target(d, frame)
    norm = (d['qInf'] * d['D'])[:, None]
    true = np.column_stack([d['Sx'], d['Sy']]) / norm
    first = d['dataset'] == 0
    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
    rms = lambda a: np.sqrt(np.mean(a ** 2))

    Sfit = trainingFit(d, X, Y, names, frame)
    Scv = heldOutPredictions(d, X, Y, names, frame)

    # Showcase case with the exported held-out net
    runNames = sorted({str(s) for s in np.load(os.path.join(
        common.DATA, 'datasets', dataFiles[0]))['runNames']})
    mShow = first & (d['run'] == np.unique(d['run'][first])[runNames.index(showcase)])
    rows = {k: v[mShow] for k, v in d.items()}
    Sshow = ft.toSource(loadNet(showcaseNet)(X[mShow]), rows, frame) / norm[mShow]

    # Cd, Cl per (snapshot, level), then mean over levels per snapshot for the plot
    g, Ctrue = coefficients(rows, true[mShow])
    _, Cnn = coefficients(rows, Sshow)
    tOf = {gg: tt for gg, tt in zip(rows['group'], rows['t'])}
    tg = np.array([tOf[gg] for gg in g])
    ts = np.unique(tg)
    mean = lambda C: np.array([C[tg == tt].mean(axis=0) for tt in ts])
    c = oracle.parseCase(showcase)
    tRef = newRun.freestream(c.mach, c.Re)['tRef']
    parity(true[first], Scv[first], mShow[first], Sshow)
    apriori(ts / tRef, mean(Ctrue), mean(Cnn))

    # Numbers
    gA, CtA = coefficients({k: v[first] for k, v in d.items()}, true[first])
    _, CcvA = coefficients({k: v[first] for k, v in d.items()}, Scv[first])
    _, CfitA = coefficients({k: v[first] for k, v in d.items()}, Sfit[first])
    typical = rms(true[first])
    with open(os.path.join(outDir, 'accuracy.txt'), 'w') as fh:
        def out(s=''):
            print(s)
            fh.write(s + '\n')
        out(f'net: input set {setName!r}, MLP {len(names)}-{"-".join(map(str, train.hidden))}-2 '
            f'tanh, --mirror, data {" + ".join(dataFiles)}')
        out(f'unsteady rows: {first.sum()} columns x snapshots x levels, '
            f'{len(np.unique(d["run"][first]))} cases; total training rows {len(first)} '
            f'({len(np.unique(d["run"]))} cases incl. steady3d)')
        out(f'typical |S/(q_inf D)| of a column (RMS): {typical:.3f}')
        out()
        out(f'{"":<34}{"cell RMSE":>10}{"rel":>7}{"Cd err RMS":>12}{"Cl err RMS":>12}')
        for label, S, C in (('training fit (all rows)', Sfit[first], CfitA),
                            ('held out by case (5-fold CV)', Scv[first], CcvA)):
            out(f'{label:<34}{rmse(S, true[first]):10.4f}{rmse(S, true[first]) / typical:7.1%}'
                f'{rms(C[:, 0] - CtA[:, 0]):12.4f}{rms(C[:, 1] - CtA[:, 1]):12.4f}')
        for c, lab in enumerate(('Sx', 'Sy')):
            a, b = Scv[first][:, c], true[first][:, c]
            out(f'held-out R^2 {lab}: {1 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2):.4f}')
        out()
        out(f'showcase {showcase}, net {showcaseNet} (case not in training):')
        out(f'  cell RMSE {rmse(Sshow, true[mShow]):.4f} ({rmse(Sshow, true[mShow]) / typical:.1%} of typical)')
        out(f'  a priori Cd err RMS over snapshots {rms(Cnn[:, 0] - Ctrue[:, 0]):.4f}, '
            f'Cl {rms(Cnn[:, 1] - Ctrue[:, 1]):.4f}')
        last = tg == ts.max()
        out(f'  final time: oracle Cd {Ctrue[last, 0].mean():.3f} NN {Cnn[last, 0].mean():.3f}; '
            f'oracle Cl {Ctrue[last, 1].mean():+.3f} NN {Cnn[last, 1].mean():+.3f}')
