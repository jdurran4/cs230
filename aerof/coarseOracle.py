"""Coarse-input training data: one coarse oracle run per resolved run.

The network trained on fine-solution averages sees different inputs online (the coarse
solver's node states): at the converged oracle state it predicts Cd 1.12 instead of 1.76
(nnCheck.py). Here every resolved run's exact binned source (grid 0, the simulation
grid, D/h = 1.2) is replayed on the coarse mesh, and the training rows are rebuilt from
the converged coarse state: inputs AND target normalization (pbar, qbar, flow frame e)
come from the coarse solver, exactly as the online source will compute them. Then the
oracle solution is a fixed point of the online system by construction.

Usage (from aerof/):
    python coarseOracle.py setup     coarse/<run>/ for every run with samples.npz
    bash coarse/runAll.sh            run + postpro all of them (16 at a time, 1 rank each)
    python coarseOracle.py build     -> dataset_coarse.npz (same columns as dataset_multi)

AERO-F binary: nn/aerof.oracle.opt, a frozen copy of the source-terms build with
OracleSourceFile (so rebuilding AERO-F meanwhile doesn't touch these runs).
"""
import glob
import os
import re
import shutil
import sys

import numpy as np

import binTargets
import newRun

here = os.path.dirname(os.path.abspath(__file__))
coarseDir = os.path.join(here, 'coarse')
template = os.path.join(here, 'oracle', 'M1.5_Re100_x0.04_y0.12')
g, R_gas, C_suth, S_suth = newRun.gamma, newRun.R_gas, newRun.C_suth, newRun.S_suth


def runNames():
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(here, 'runs', '*', 'samples.npz')))


def setup():
    text = open(os.path.join(template, 'FluidFile')).read()
    text = text.split('under Problem', 1)[1]                    # drop the case header
    for name in runNames():
        d = os.path.join(coarseDir, name)
        os.makedirs(d, exist_ok=True)
        mach, rho, p = binTargets.readInlet(os.path.join(here, 'runs', name, 'FluidFile'))
        head, inlet, tail = re.split(r'(under\s+Inlet\s*\{.*?\})', text, maxsplit=1,
                                     flags=re.S)                 # only the Inlet block
        inlet = re.sub(r'(Mach\s*=\s*)[^;]+;', rf'\g<1>{mach:.10g};', inlet)
        inlet = re.sub(r'(Density\s*=\s*)[^;]+;', rf'\g<1>{rho:.10e};', inlet)
        inlet = re.sub(r'(Pressure\s*=\s*)[^;]+;', rf'\g<1>{p:.10e};', inlet)
        t = head + inlet + tail
        header = (f'// Coarse oracle run for runs/{name} (written by coarseOracle.py from\n'
                  f'// oracle/M1.5_Re100_x0.04_y0.12/FluidFile; only the Inlet differs).\n\n')
        open(os.path.join(d, 'FluidFile'), 'w').write(header + 'under Problem' + t)
        shutil.copy(os.path.join(template, 'postpro.sh'), d)   # it cd's to its own dir
        s = np.load(os.path.join(here, 'runs', name, 'samples.npz'))
        g0 = s['grid'] == 0
        f = np.column_stack([s['Sx'][g0], s['Sy'][g0]]) / (s['h'][g0] ** 2)[:, None]
        np.savetxt(os.path.join(d, 'oracleSource.txt'),
                   np.column_stack([s['colX'][g0], s['colY'][g0], f]), fmt='%.16e')
    with open(os.path.join(coarseDir, 'runAll.sh'), 'w') as f:
        f.write("""#!/bin/bash
# Run every coarse oracle case (written by ../coarseOracle.py setup), 16 at a time.
cd "$(dirname "$0")"
source ../tools.sh
BIN=$(cd .. && pwd)/nn/aerof.oracle.opt
one() {
    cd "$1" || return
    mkdir -p results references postpro
    mpirun -np 1 "$BIN" FluidFile > log.out 2>&1
    bash postpro.sh > postpro.log 2>&1
    last=$(grep -E '^It ' log.out | tail -1)
    echo "$1 $last"
}
export -f one; export BIN
ls -d */ | sed 's#/##' | xargs -P 16 -I{} bash -c 'one {}' > runAll.log
echo "all done: $(wc -l < runAll.log) runs" >> runAll.log
""")
    print(f'{len(runNames())} coarse cases in {coarseDir}')


def readXpostLast(path):
    from binInputs import readXpost
    return readXpost(path)[1][-1]


def build():
    import binOneCell as bin1
    raw = np.load(os.path.join(here, 'dataset_multi.npz'))
    names = list(raw['featureNames'])
    multi = {k: raw[k] for k in names}
    rows = {k: [] for k in names}
    nodes = colOf = None
    bad = []
    for name in runNames():
        d = os.path.join(coarseDir, name)
        post = os.path.join(d, 'postpro')
        log = open(os.path.join(d, 'log.out')).read() if os.path.isfile(os.path.join(d, 'log.out')) else ''
        its = re.findall(r'^It\s+\d+: Res = (\S+),', log, re.M)
        if not its or float(its[-1]) > 1e-5 or 'Aborting' in log or not os.path.isdir(post):
            bad.append(name)
            continue
        if nodes is None:
            nodes, _ = bin1.readDomain(os.path.join(post, 'OUTPUT.top.0000'))
            cols, colOf = np.unique(np.round(nodes[:, :2], 9), axis=0, return_inverse=True)
            colOf = colOf.ravel()
            cnt = np.bincount(colOf)
        cm = lambda a: np.bincount(colOf, a) / cnt
        rho = cm(readXpostLast(os.path.join(post, 'Density.0000.xpost')))
        vel = readXpostLast(os.path.join(post, 'Velocity.0000.xpost'))
        u, v = cm(vel[:, 0]), cm(vel[:, 1])
        p = cm(readXpostLast(os.path.join(post, 'Pressure.0000.xpost')))

        s = np.load(os.path.join(here, 'runs', name, 'samples.npz'))
        g0 = np.flatnonzero(s['grid'] == 0)
        for k in g0:
            c = np.argmin(np.hypot(cols[:, 0] - s['colX'][k], cols[:, 1] - s['colY'][k]))
            speed = np.hypot(u[c], v[c])
            e = np.array([u[c], v[c]]) / speed
            ep = np.array([-e[1], e[0]])
            T = p[c] / (rho[c] * R_gas)
            mu = C_suth * T ** 1.5 / (T + S_suth)
            # geometry back to x-y with the FINE frame it was stored in
            ef = np.array([s['ePar_x'][k], s['ePar_y'][k]])
            efp = np.array([-ef[1], ef[0]])
            N = s['Npar'][k] * ef + s['Nperp'][k] * efp
            r = (s['rpar'][k] * ef + s['rperp'][k] * efp) * s['h'][k]
            rn = np.linalg.norm(r)
            S = np.array([s['Sx'][k], s['Sy'][k]])
            D = s['D'][k]
            pB, qB = p[c], 0.5 * rho[c] * speed ** 2
            Sp = S - pB * D * N
            new = dict(
                Mbar=speed / np.sqrt(g * p[c] / rho[c]), ReD=rho[c] * speed * D / mu,
                Npar=N @ e, Nperp=N @ ep, cosT=r @ e / rn, sinT=r @ ep / rn,
                rpar=r @ e / s['h'][k], rperp=r @ ep / s['h'][k], pqRatio=pB / qB,
                Spar_q=Sp @ e / (qB * D), Sperp_q=Sp @ ep / (qB * D),
                Spar_pq=Sp @ e / ((pB + qB) * D), Sperp_pq=Sp @ ep / ((pB + qB) * D),
                SFpar_q=S @ e / (qB * D), SFperp_q=S @ ep / (qB * D),
                SFpar_pq=S @ e / ((pB + qB) * D), SFperp_pq=S @ ep / ((pB + qB) * D),
                pBar=pB, qBar=qB, ePar_x=e[0], ePar_y=e[1])
            # everything else (geometry, SxInf, meta, run id) as in dataset_multi
            j = np.flatnonzero((multi['grid'] == 0) & np.isclose(multi['colX'], s['colX'][k])
                               & np.isclose(multi['colY'], s['colY'][k])
                               & np.isclose(multi['machInf'], s['machInf'][k])
                               & np.isclose(multi['ReInf'], s['ReInf'][k])
                               & np.isclose(multi['ox'], s['ox'][k])
                               & np.isclose(multi['oy'], s['oy'][k]))
            src = {kk: multi[kk][j[0]] for kk in names} if len(j) else \
                {kk: (s[kk][k] if kk in s.files else np.nan) for kk in names}
            for kk in names:
                rows[kk].append(new.get(kk, src[kk]))
    out = {k: np.array(v) for k, v in rows.items()}
    np.savez(os.path.join(here, 'dataset_coarse.npz'), featureNames=np.array(names), **out)
    print(f'{len(runNames()) - len(bad)} runs, {len(out["Mbar"])} rows -> dataset_coarse.npz; '
          f'{len(bad)} not converged/missing: {bad[:10]}')


if __name__ == '__main__':
    {'setup': setup, 'build': build}[sys.argv[1]]()
