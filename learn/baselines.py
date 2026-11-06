"""Baselines for the source-term MLP, scored exactly like learn/train.py's CV (same rows,
same folds by case, same mirror augmentation, same metrics), by swapping the model:

  zero    the network output is 0, so S = (pbar - pRef) D N: the physics baseline (the
          pressure of the local coarse flow on the cell's piece of surface), no learning
  linear  least squares on the standardized inputs (with an intercept), per output

Usage (from the repo root), same data options as train.py:
  python -m learn.baselines [--mirror] [--exclude NAME] --data FILE [--data FILE ...]
"""
import sys

import numpy as np

from learn import train


def fitZero(X, Y, seed, W=None):
    return 'zero', None


def fitLinear(X, Y, seed, W=None):
    xMean, xStd = X.mean(axis=0), X.std(axis=0)
    xStd[xStd < 1e-8 * np.maximum(np.abs(xMean), 1.0)] = 1.0   # constant input (Dh)
    A = np.column_stack([(X - xMean) / xStd, np.ones(len(X))])  # inputs + intercept
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)                # (n_in + 1, 2)
    return 'linear', (xMean, xStd, coef)


def predictBaseline(model, stats, X):
    if model == 'zero':
        return np.zeros((len(X), 2))
    xMean, xStd, coef = stats
    return np.column_stack([(X - xMean) / xStd, np.ones(len(X))]) @ coef


if __name__ == '__main__':
    args = sys.argv[1:]
    train.mirror = '--mirror' in args
    args = [a for a in args if a != '--mirror']
    while '--data' in args:
        i = args.index('--data')
        train.dataFiles.append(args[i + 1])
        del args[i:i + 2]
    while '--exclude' in args:
        i = args.index('--exclude')
        train.exclude.append(args[i + 1])
        del args[i:i + 2]
    train.seeds = [0]                         # deterministic models: one fold shuffle
    train.predict = predictBaseline
    for name, fit in (('zero', fitZero), ('linear', fitLinear)):
        print(f'==== baseline: {name}')
        train.fit = fit
        train.cv(['xy'])
