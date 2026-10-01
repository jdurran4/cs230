"""
Phase 4 targets: momentum source term per coarse control volume, per output step.

Pipeline (see README):
  1-2. EmbeddedSurfaceForce (force on the body, p - p_inf + shear) summed over each x-y
       station (the bottom and top ring nodes)
  3-5. bin into coarse median-dual columns with binOneCell.columnFractions
  6.   restore absolute pressure: F_abs(K) = F(K) - p_inf * sum_s frac[s,K] * (n ds)_s * dz
       where (n ds)_s is the station's tent-weighted outward normal of the discrete
       surface: half of each neighboring segment's length x outward normal
  7.   source on the fluid: S(K) = -F_abs(K)
  8.   per unit span: divide by dz  [N/m];  also nondimensional S / (q_inf D)

Freestream p, rho and Mach are read from the run's FluidFile (BoundaryConditions/Inlet).

Writes runs/<run>/targets.npz with
  times (nt,), columns (ncol, 2) grid-point x-y, S (nt, ncol, 2) [N/m],
  Snd (nt, ncol, 2) = S / (q D), Srel (nt, ncol, 2) = the same without the p_inf term,
  pInf, rhoInf, mach, U, q, D, dz

Usage: set runNames below (None = every run with postpro/EmbeddedSurfaceForce), then run.
"""
import os
import re

import numpy as np

import binOneCell as bin1
import checkSurfaceForce as chk
import cylMeshGen as mesh
import newRun

# ============= User-defined parameters ==============================================
runNames = None
nzSample = 64              # z samples per station for the dual-cell footprint

aerofDir = os.path.dirname(os.path.abspath(__file__))
runsDir = os.path.join(aerofDir, 'runs')


# ============= Helpers ==============================================================
def readInlet(fluidFile):
    """Mach, Density, Pressure from the BoundaryConditions/Inlet block."""
    text = open(fluidFile).read()
    block = re.search(r'under\s+Inlet\s*\{(.*?)\}', text, re.S).group(1)
    get = lambda key: float(re.search(rf'\b{key}\s*=\s*([^;]+);', block).group(1))
    return get('Mach'), get('Density'), get('Pressure')


def stationNormals(xy):
    """Tent-weighted outward normal x arc length of each station of a closed polygon,
    per unit span: (n ds)_s = 0.5 * (L n)_prev + 0.5 * (L n)_next.  (ns, 2)"""
    cen = xy.mean(axis=0)
    order = np.argsort(np.arctan2(xy[:, 1] - cen[1], xy[:, 0] - cen[0]))   # CCW
    p = xy[order]
    seg = np.roll(p, -1, axis=0) - p                         # segment k: p[k] -> p[k+1]
    segN = np.column_stack([seg[:, 1], -seg[:, 0]])          # outward for CCW, |.| = L
    nds = np.empty_like(xy)
    nds[order] = 0.5 * (segN + np.roll(segN, 1, axis=0))
    return nds


def binTargets(runName, nodes, tets, X):
    runDir = os.path.join(runsDir, runName)
    mach, rho, pInf = readInlet(os.path.join(runDir, 'FluidFile'))
    U = mach * np.sqrt(newRun.gamma * pInf / rho)
    q = 0.5 * rho * U ** 2
    D, dz = mesh.D, mesh.dz

    tF, Fs = chk.readSurfaceForce(os.path.join(runDir, 'postpro', 'EmbeddedSurfaceForce'))
    xy, station = np.unique(np.round(X[:, :2], 12), axis=0, return_inverse=True)
    Fst = np.zeros((len(tF), len(xy), 2))
    np.add.at(Fst, (slice(None), station.ravel()), Fs[:, :, :2])            # (nt, ns, 2)

    frac, columns = bin1.columnFractions(xy, nodes, tets, nzSample)
    nds = stationNormals(xy)
    Frel = np.einsum('sc,tsd->tcd', frac, Fst)                              # (nt, ncol, 2)
    Fpinf = -pInf * dz * (frac.T @ nds)                                     # (ncol, 2)
    S = -(Frel + Fpinf) / dz
    Srel = -Frel / dz

    out = os.path.join(runDir, 'targets.npz')
    np.savez(out, times=tF, columns=columns, S=S, Snd=S / (q * D), Srel=Srel,
             pInf=pInf, rhoInf=rho, mach=mach, U=U, q=q, D=D, dz=dz)

    # checks / summary
    tAvg = tF > 0.5 * tF[-1]
    print(f'\n{runName}: M = {mach:g}, p_inf = {pInf:.4g} Pa, q = {q:.4g} Pa, '
          f'p_inf/q = {pInf / q:.3f};  {len(tF)} steps, {len(columns)} columns')
    print(f'   closed surface: |sum_s n ds| = {np.abs(nds.sum(axis=0)).max():.1e} m '
          f'(perimeter {np.linalg.norm(nds, axis=1).sum():.4f}, pi D = {np.pi * D:.4f})')
    print(f'   sum_K S - (-sum_nodes Fs / dz), max over time = '
          f'{np.abs(S.sum(axis=1) + Fs[:, :, :2].sum(axis=1) / dz).max():.1e} N/m')
    print(f'   2nd-half average, S on the FLUID per unit span [N/m] and S / (q D):')
    print(f'   {"grid point x, y":>18}  {"Sx":>10}  {"Sy":>10}  {"Sx (p-pinf)":>11}  '
          f'{"Sy (p-pinf)":>11}  {"Sx/qD":>7}  {"Sy/qD":>7}')
    Sa, Sra = S[tAvg].mean(axis=0), Srel[tAvg].mean(axis=0)
    for c in np.argsort(-np.linalg.norm(Sa, axis=1)):
        print(f'   {columns[c, 0]:8.3f} {columns[c, 1]:8.3f}  {Sa[c, 0]:10.3e}  {Sa[c, 1]:10.3e}  '
              f'{Sra[c, 0]:11.3e}  {Sra[c, 1]:11.3e}  {Sa[c, 0] / (q * D):7.3f}  '
              f'{Sa[c, 1] / (q * D):7.3f}')
    print(f'   total Sx/qD = {Sa[:, 0].sum() / (q * D):.3f}  (= -Cd)')
    print(f'   wrote {out}')


# ============= Main =================================================================
if __name__ == '__main__':
    names = runNames or sorted(
        r for r in os.listdir(runsDir)
        if os.path.isfile(os.path.join(runsDir, r, 'postpro', 'EmbeddedSurfaceForce')))
    nodes, tets = bin1.readDomain(bin1.domainFile)
    X = chk.readSurfaceNodes(chk.surfFile)
    for name in names:
        binTargets(name, nodes, tets, X)
