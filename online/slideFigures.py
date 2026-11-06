"""Slide figures for the held-out M1.5 online test: the resolved run, its coarse oracle,
and online runs with the steady and the combined (unsteady + steady3d) networks.

  forces.pdf  Cd(t), Cl(t). Resolved: the embedded-surface force at every time step
              (LiftandDrag / (q_inf D span)). Oracle: the binned source it applies, dashed on
              top (it is the resolved force by construction). Online: the network's source.
  wake.pdf    u/U along lines across the wake at x = xc + 10, 20, 30 D (wakeNear.pdf:
              2.5, 5, 10 D, all inside the resolved run's AMR region), y = yc +- 5 D, at
              the final time. Coarse runs: their node values on the nearest node column (at
              the interior z-level). Resolved: interpolated from its last AMR mesh along the
              same x. Freestream (u/U = 1) is what a coarse run without the cable gives.

  forcesMeshes.pdf  Cd(t), Cl(t) of the per-mesh 64x64 networks (M1.5 held out) online on
              each coarse mesh D/h = 4.8, 2.4, 1.2, 0.6 (the rows of the ParaView movie),
              with the resolved run and its oracle (D/h 1.2, dashed). A run that stopped
              early ends with an x.

Usage (from the repo root):  python -m online.slideFigures [meshes]
Writes DATA/unsteady/online/slides/{forces,wake,wakeNear}.{pdf,png}, or with `meshes`
only forcesMeshes.{pdf,png}.
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import common
from online import unsteady
from sweep import meshGen, newRun, oracle

name = 'M1.5_Re100_x0.04_y0.12'
nets = {'steadyNet': 'steady NN (M1.5 in training)',
        'xy_comb_noM15': 'unsteady + steady NN (M1.5 held out)'}
wakeX = {'wake': (10, 20, 30),     # line positions downstream of the cylinder center [D]
         'wakeNear': (2.5, 5, 10)}  # inside the resolved run's AMR region (ends ~12 D
                                    # downstream; beyond it the resolved run is on the
                                    # base mesh, h = D / 1.2 like the coarse runs)
wakeHalfWidth = 5                 # lines span yc +- this [D]

grey = '#52514e'
style = {'resolved': dict(color=grey, lw=2.2),
         'oracle': dict(color=unsteady.colors[0], lw=1.8, ls='--'),
         'steadyNet': dict(color=unsteady.colors[1], lw=1.8),
         'xy_comb_noM15': dict(color=unsteady.colors[2], lw=1.8)}
# per-mesh runs: categorical slots 2-5 (slot 1, blue, is the oracle)
meshColors = [unsteady.colors[1], unsteady.colors[2], unsteady.colors[3], '#8a5cd6']
outDir = os.path.join(common.UNSTEADY, 'online', 'slides')


def tidy(ax):
    ax.grid(color='#e1e0d9', lw=0.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def save(fig, stem):
    os.makedirs(outDir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outDir, f'{stem}.{ext}'), dpi=200, bbox_inches='tight')
    print('wrote', os.path.join(outDir, stem + '.{pdf,png}'))


# ============= Cd, Cl ===============================================================
def forces():
    fs, curves = unsteady.forceCurves(name, list(nets))
    resolvedDir = os.path.join(newRun.runsDir, name)
    geo = meshGen.loadGeometry(resolvedDir)
    hist = np.loadtxt(os.path.join(resolvedDir, 'postpro', 'LiftandDrag'))
    qDs = 0.5 * fs['rho'] * fs['U'] ** 2 * geo['D'] * geo['span']
    tR, CdR, ClR = hist[:, 1] / fs['tRef'], hist[:, 4] / qDs, hist[:, 5] / qDs
    print(f'resolved Cd, Cl at the end {CdR[-1]:.3f} {ClR[-1]:+.3f}; '
          f'oracle {curves["oracle (true)"][1][-1]:.3f} {curves["oracle (true)"][2][-1]:+.3f}')

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, q, R in ((ax[0], 1, CdR), (ax[1], 2, ClR)):
        a.plot(tR, R, label='resolved', **style['resolved'])
        c = curves['oracle (true)']
        a.plot(c[0] / fs['tRef'], c[q], label='oracle (applied source)', **style['oracle'])
        for tag, lbl in nets.items():
            c = curves[tag]
            a.plot(c[0] / fs['tRef'], c[q], label=lbl, **style[tag])
        a.set_xlabel('t  [D/U]')
        tidy(a)
    ax[0].set_ylabel('$C_d$')
    ax[1].set_ylabel('$C_l$')
    ax[0].set_ylim(bottom=0)
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, 'forces')


# ============= Cd, Cl per mesh ======================================================
meshes = (4.8, 2.4, 1.2, 0.6)        # finest first, as in the movie
oracleMesh = 1.2


def forcesMeshes():
    curves = {}
    for Dh in meshes:
        oracle.setMesh(Dh)
        fs, c = unsteady.forceCurves(name, [f'xy_Dh{Dh:g}_h64_held4'])
        curves[Dh] = c[f'xy_Dh{Dh:g}_h64_held4']
        if Dh == oracleMesh:
            curves['oracle'] = c['oracle (true)']
    resolvedDir = os.path.join(newRun.runsDir, name)
    geo = meshGen.loadGeometry(resolvedDir)
    hist = np.loadtxt(os.path.join(resolvedDir, 'postpro', 'LiftandDrag'))
    qDs = 0.5 * fs['rho'] * fs['U'] ** 2 * geo['D'] * geo['span']
    tR, CdR, ClR = hist[:, 1] / fs['tRef'], hist[:, 4] / qDs, hist[:, 5] / qDs
    tEnd = tR[-1]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, q, R in ((ax[0], 1, CdR), (ax[1], 2, ClR)):
        a.plot(tR, R, label='resolved (HDM)', **style['resolved'])
        c = curves['oracle']
        a.plot(c[0] / fs['tRef'], c[q], label=f'oracle (D/h {oracleMesh:g})', **style['oracle'])
        for m, Dh in enumerate(meshes):
            c = curves[Dh]
            t = c[0] / fs['tRef']
            col = meshColors[m]
            a.plot(t, c[q], color=col, lw=1.6, label=f'NN online, D/h {Dh:g}')
            if t[-1] < tEnd - 1.0:
                a.plot(t[-1], c[q][-1], 'x', color=col, ms=8, mew=2)
                print(f'D/h {Dh:g} stops at t = {t[-1]:.1f} D/U')
        a.set_xlabel('t  [D/U]')
        tidy(a)
    ax[0].set_ylabel('$C_d$')
    ax[1].set_ylabel('$C_l$')
    ax[0].set_ylim(bottom=0)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    save(fig, 'forcesMeshes')
    for Dh in meshes:
        c = curves[Dh]
        print(f'D/h {Dh:g}: Cd, Cl at t_end {c[1][-1]:.3f} {c[2][-1]:+.3f}')
    print(f'resolved: {CdR[-1]:.3f} {ClR[-1]:+.3f}')


# ============= Wake profiles ========================================================
def lastFrame(d):
    """Times-last state of the resolved run: nodes of its last AMR mesh and u there."""
    tops = sorted(glob.glob(os.path.join(d, 'postpro', 'OUTPUT.top.[0-9][0-9][0-9][0-9]')))
    k = tops[-1].rsplit('.', 1)[1]
    nodes, _ = common.readTop(tops[-1])
    t, vel = common.readXpost(os.path.join(d, 'postpro', f'Velocity.{k}.xpost'))
    return t[-1], nodes, vel[-1, :, 0]


def wake(xs, stem):
    c = oracle.parseCase(name)
    fs = newRun.freestream(c.mach, c.Re)
    D = meshGen.D
    xc, yc = newRun.cylinderCenter(c)

    # resolved: linear interpolation on the triangulated z = 0 plane of the last mesh
    tR, nodesR, uR = lastFrame(os.path.join(newRun.runsDir, name))
    plane = np.abs(nodesR[:, 2] - nodesR[:, 2].min()) < 1e-9
    interp = mtri.LinearTriInterpolator(
        mtri.Triangulation(nodesR[plane, 0], nodesR[plane, 1]), uR[plane] / fs['U'])

    # coarse runs: node values at the interior z-level, last snapshot
    nodes, _ = common.readTop(oracle.meshFile)
    on = np.rint(nodes[:, 2] / oracle.hz).astype(int) == unsteady.level
    coarse = {'oracle': os.path.join(oracle.oracleDir, 'runs', name)}
    coarse.update({tag: os.path.join(common.UNSTEADY, 'online', f'{name}_{tag}') for tag in nets})
    final = {}
    for key, d in coarse.items():
        t, f = unsteady.fields(d)
        final[key] = f['u'][-1] / fs['U']
        print(f'{key}: t_end = {t[-1] / fs["tRef"]:.2f} D/U')
    print(f'resolved: t_end = {tR / fs["tRef"]:.2f} D/U')

    fig, ax = plt.subplots(1, len(xs), figsize=(10, 4.2), sharey=True)
    for a, dx in zip(ax, xs):
        cols = np.unique(np.round(nodes[on, 0], 9))
        x = cols[np.argmin(np.abs(cols - (xc + dx * D)))]           # nearest node column
        sel = np.flatnonzero(on & (np.abs(nodes[:, 0] - x) < 1e-6)
                             & (np.abs(nodes[:, 1] - yc) <= wakeHalfWidth * D + 1e-9))
        sel = sel[np.argsort(nodes[sel, 1])]
        y = (nodes[sel, 1] - yc) / D

        yf = np.linspace(-wakeHalfWidth, wakeHalfWidth, 801)
        a.plot(interp(np.full_like(yf, x), yc + yf * D), yf, label='resolved',
               **style['resolved'])
        a.plot(final['oracle'][sel], y, marker='o', ms=4, label='oracle', **style['oracle'])
        for tag, lbl in nets.items():
            a.plot(final[tag][sel], y, marker='o', ms=4, label=lbl, **style[tag])
        a.axvline(1.0, color='#b5b3ab', lw=1, ls=':', label='freestream (no cable)')
        a.set_title(f'x = {(x - xc) / D:.1f} D behind the cylinder', fontsize=10)
        a.set_xlabel('u / U')
        tidy(a)
    ax[0].set_ylabel('y / D')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, stem)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['meshes']:
        forcesMeshes()
        sys.exit()
    forces()
    for stem, xs in wakeX.items():
        wake(xs, stem)
