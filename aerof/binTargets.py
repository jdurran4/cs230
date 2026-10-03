"""
Phase 4 targets: momentum source term per coarse control volume, per output step.

Pipeline (see README):
  1-2. EmbeddedSurfaceForce (force on the body, p - p_inf + shear) summed over each x-y
       station (the bottom and top ring nodes)
  3-5. bin into coarse median-dual columns with binOneCell.columnFractions
  6.   restore absolute pressure: F_abs(K) = F(K) - p_inf * sum_s frac[s,K] * (n ds)_s * span
       where (n ds)_s is the station's tent-weighted outward normal of the discrete
       surface: half of each neighboring segment's length x outward normal
  7.   source on the fluid: S(K) = -F_abs(K)
  8.   per unit span: divide by span  [N/m];  also nondimensional S / (q_inf D)
  9.   local (what the coarse solver has online), from the cell averages in inputs.npz
       (run binInputs.py first):
         N(K)  = sum_s frac[s,K] (n ds)_s / D   area-weighted outward normal of the body
                 piece in K per unit span, over D  (N = 0 for a whole closed body)
         Sp(K) = S(K) - pbar(K) D N(K)          source with the cell-average pressure
                 pbar as reference, so S = Sp + pbar D N exactly
         flow frame e_par = ubar/|ubar|, e_perp = z x e_par; qbar = rhobar |ubar|^2 / 2
         Npar, Nperp = N.e_par, N.e_perp;  Spar, Sperp = Sp.e_par, Sp.e_perp / (qbar D)
       Back to x-y: S = qbar D (Spar e_par + Sperp e_perp) + pbar D N.

span is the embedded tube length (1 + 2 zOverhang) dz, not the fluid slab thickness dz:
AERO-F integrates the force over the whole tube, overhang included (the stagnation-node
estimate in checkSurfaceForce.py gives 0.153 m for the 0.150 m tube at dz = 0.1, Cd 1.76
for M1.5 Re100; the zOverhang = 0.1 case tests it). Geometry (center, dz, zOverhang,
surface file) is per case: cylMeshGen.loadGeometry(runs/<run>).

Freestream p, rho and Mach are read from the run's FluidFile (BoundaryConditions/Inlet).

Writes runs/<run>/targets.npz with
  times (nt,), columns (ncol, 2) grid-point x-y, S (nt, ncol, 2) [N/m],
  Snd (nt, ncol, 2) = S / (q D), Srel (nt, ncol, 2) = the same without the p_inf term,
  pInf, rhoInf, mach, U, q, D, dz, span, xc, yc,
  N (ncol, 2), ePar (nt, ncol, 2), pBar, qBar (nt, ncol), Sp (nt, ncol, 2) [N/m],
  Npar, Nperp, Spar, Sperp (nt, ncol)        local flow frame, nondimensional

Steady runs give one sample (nt = 1): the converged solution.

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


def caseDir(runName):
    return os.path.join(runsDir, runName)


def geometry(runName):
    return mesh.loadGeometry(caseDir(runName))


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


def binTargets(runName, nodes, tets):
    runDir = caseDir(runName)
    mach, rho, pInf = readInlet(os.path.join(runDir, 'FluidFile'))
    U = mach * np.sqrt(newRun.gamma * pInf / rho)
    q = 0.5 * rho * U ** 2
    geo = geometry(runName)
    D, dz, span = geo['D'], geo['dz'], geo['span']
    X = chk.readSurfaceNodes(geo['surfFile'])

    tF, Fs = chk.readSurfaceForce(os.path.join(runDir, 'postpro', 'EmbeddedSurfaceForce'))
    xy, station = np.unique(np.round(X[:, :2], 12), axis=0, return_inverse=True)
    Fst = np.zeros((len(tF), len(xy), 2))
    np.add.at(Fst, (slice(None), station.ravel()), Fs[:, :, :2])            # (nt, ns, 2)

    frac, columns = bin1.columnFractions(xy, nodes, tets, nzSample)
    nds = stationNormals(xy)
    Frel = np.einsum('sc,tsd->tcd', frac, Fst)                              # (nt, ncol, 2)
    Fpinf = -pInf * span * (frac.T @ nds)                                   # (ncol, 2)
    S = -(Frel + Fpinf) / span
    Srel = -Frel / span

    # local flow frame from the cell averages (step 9)
    N = frac.T @ nds / D                                                    # (ncol, 2)
    inp = np.load(os.path.join(runDir, 'inputs.npz'))
    col = np.array([np.argmin(np.linalg.norm(inp['columns'] - c, axis=1)) for c in columns])
    assert np.allclose(inp['columns'][col], columns), 'target columns not in inputs.npz'
    ti = np.array([np.argmin(np.abs(inp['times'] - t)) for t in tF])
    assert np.allclose(inp['times'][ti], tF, rtol=1e-5), 'output times differ from inputs.npz'
    u = inp['u'][ti][:, col]                                                # (nt, ncol, 2)
    pBar = inp['p'][ti][:, col]
    qBar = 0.5 * inp['rho'][ti][:, col] * (u ** 2).sum(axis=2)
    ePar = u / np.linalg.norm(u, axis=2, keepdims=True)
    ePerp = np.stack([-ePar[..., 1], ePar[..., 0]], axis=2)
    Sp = S - pBar[..., None] * D * N
    Npar, Nperp = (N * ePar).sum(axis=2), (N * ePerp).sum(axis=2)
    Spar, Sperp = ((Sp * e).sum(axis=2) / (qBar * D) for e in (ePar, ePerp))

    out = os.path.join(runDir, 'targets.npz')
    np.savez(out, times=tF, columns=columns, S=S, Snd=S / (q * D), Srel=Srel,
             pInf=pInf, rhoInf=rho, mach=mach, U=U, q=q, D=D, dz=dz, span=span,
             xc=geo['xc'], yc=geo['yc'], N=N, ePar=ePar, pBar=pBar, qBar=qBar, Sp=Sp,
             Npar=Npar, Nperp=Nperp, Spar=Spar, Sperp=Sperp)

    # checks / summary
    tAvg = tF > 0.5 * tF[-1]
    print(f'\n{runName}: M = {mach:g}, p_inf = {pInf:.4g} Pa, q = {q:.4g} Pa, '
          f'p_inf/q = {pInf / q:.3f};  {len(tF)} steps, {len(columns)} columns;  '
          f'center ({geo["xc"]:.4f}, {geo["yc"]:.4f}), span {span:.4g} m (dz {dz:g})')
    print(f'   closed surface: |sum_s n ds| = {np.abs(nds.sum(axis=0)).max():.1e} m '
          f'(perimeter {np.linalg.norm(nds, axis=1).sum():.4f}, pi D = {np.pi * D:.4f})')
    print(f'   sum_K S - (-sum_nodes Fs / span), max over time = '
          f'{np.abs(S.sum(axis=1) + Fs[:, :, :2].sum(axis=1) / span).max():.1e} N/m')
    print(f'   2nd-half average, S on the FLUID per unit span [N/m] and S / (q D):')
    print(f'   {"grid point x, y":>18}  {"Sx":>10}  {"Sy":>10}  {"Sx (p-pinf)":>11}  '
          f'{"Sy (p-pinf)":>11}  {"Sx/qD":>7}  {"Sy/qD":>7}')
    Sa, Sra = S[tAvg].mean(axis=0), Srel[tAvg].mean(axis=0)
    for c in np.argsort(-np.linalg.norm(Sa, axis=1)):
        print(f'   {columns[c, 0]:8.3f} {columns[c, 1]:8.3f}  {Sa[c, 0]:10.3e}  {Sa[c, 1]:10.3e}  '
              f'{Sra[c, 0]:11.3e}  {Sra[c, 1]:11.3e}  {Sa[c, 0] / (q * D):7.3f}  '
              f'{Sa[c, 1] / (q * D):7.3f}')
    print(f'   total Sx/qD = {Sa[:, 0].sum() / (q * D):.3f}  (= -Cd)')
    back = (qBar * D)[..., None] * (Spar[..., None] * ePar + Sperp[..., None] * ePerp) \
        + pBar[..., None] * D * N
    print(f'   local frame: sum_K N = {np.abs(N.sum(axis=0)).max():.1e}, '
          f'max |S - rotated back| = {np.abs(back - S).max():.1e} N/m')
    print(f'   {"grid point x, y":>18}  {"Npar":>6}  {"Nperp":>6}  {"Spar":>7}  {"Sperp":>7}  '
          f'{"pbar/pinf":>9}  {"qbar/q":>6}')
    for c in np.argsort(-np.linalg.norm(Sa, axis=1)):
        print(f'   {columns[c, 0]:8.3f} {columns[c, 1]:8.3f}  {Npar[-1, c]:6.3f}  '
              f'{Nperp[-1, c]:6.3f}  {Spar[-1, c]:7.3f}  {Sperp[-1, c]:7.3f}  '
              f'{pBar[-1, c] / pInf:9.3f}  {qBar[-1, c] / q:6.3f}')
    print(f'   wrote {out}')


# ============= Main =================================================================
if __name__ == '__main__':
    names = runNames or sorted(
        r for r in os.listdir(runsDir)
        if os.path.isfile(os.path.join(caseDir(r), 'postpro', 'EmbeddedSurfaceForce')))
    nodes, tets = bin1.readDomain(bin1.domainFile)
    for name in names:
        binTargets(name, nodes, tets)
