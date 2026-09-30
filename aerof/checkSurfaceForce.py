"""
Phase 3 check of the EmbeddedSurfaceForce output (per-structure-node force Fs).

For one run directory this
  1. reads postpro/EmbeddedSurfaceForce (header, node count, then per output step a
     time line and one "Fx Fy Fz" line per structure node) and postpro/LiftandDrag,
  2. sums the nodal forces at each output time and compares with LiftandDrag at the
     same time (they should agree to print precision: LiftandDrag is sum(Fs)),
  3. checks the z-force is ~0 (quasi-2D),
  4. estimates the effective span the forces are integrated over, from the force on the
     stagnation-point nodes vs. the normal-shock stagnation pressure. The embedded tube
     overhangs the fluid slab (cylMeshGen.zOverhang), so the span may be the full tube
     length (1 + 2 zOverhang) dz rather than the slab thickness dz.

Both files are dimensional (SI). Fs is the force ON the body with p - p_inf.

Usage: set runName below, then run the script (e.g. play in VS Code).
"""
import math
import os

import numpy as np

import cylMeshGen as mesh
import newRun

# ============= User-defined parameters ==============================================
runName = 'M1.5_Re100'
mach, Re = 1.5, 100        # must match the run (for the stagnation pressure estimate)

aerofDir = os.path.dirname(os.path.abspath(__file__))
runDir = os.path.join(aerofDir, 'runs', runName)
forceFile = os.path.join(runDir, 'postpro', 'EmbeddedSurfaceForce')
liftFile = os.path.join(runDir, 'postpro', 'LiftandDrag')
surfFile = os.path.join(aerofDir, 'sources', 'embeddedSurface.top')


# ============= Readers ==============================================================
def readSurfaceForce(path):
    """Returns times (nt,) and forces (nt, nn, 3)."""
    with open(path) as f:
        f.readline()                  # "Vector Force under load for ..."
        nn = int(f.readline())
        lines = f.read().split('\n')
    nt = len(lines) // (nn + 1)       # drops a trailing partial step, if any
    times = np.empty(nt)
    forces = np.empty((nt, nn, 3))
    for k in range(nt):
        step = lines[k * (nn + 1):(k + 1) * (nn + 1)]
        times[k] = float(step[0])
        forces[k] = np.array([r.split() for r in step[1:]], dtype=float)
    return times, forces


def readLiftandDrag(path):
    """Returns times (nt,) and forces (nt, 3) = (Lx, Ly, Lz)."""
    data = np.loadtxt(path, comments='#')
    return data[:, 1], data[:, 4:7]


def readSurfaceNodes(path):
    nodes = []
    with open(path) as f:
        f.readline()                  # "Nodes ..."
        for line in f:
            s = line.split()
            if s[0] == 'Elements':
                break
            nodes.append([float(x) for x in s[1:4]])
    return np.array(nodes)


# ============= Checks ===============================================================
if __name__ == '__main__':
    tF, Fs = readSurfaceForce(forceFile)
    tL, L = readLiftandDrag(liftFile)
    X = readSurfaceNodes(surfFile)
    print(f'{Fs.shape[0]} output steps, {Fs.shape[1]} structure nodes, '
          f'{len(tL)} LiftandDrag lines')

    # 1. sum(Fs) vs LiftandDrag at matching times
    Fsum = Fs.sum(axis=1)
    idx = np.clip(np.searchsorted(tL, tF), 1, len(tL) - 1)
    idx -= np.abs(tL[idx - 1] - tF) < np.abs(tL[idx] - tF)   # nearest LiftandDrag time
    match = np.abs(tL[idx] - tF) < 0.5 * np.diff(tL).min()   # within half a time step
    diff = Fsum[match] - L[idx[match]]
    scale = np.abs(L[idx[match]][:, :2]).max()
    print(f'\n1. sum(Fs) vs LiftandDrag ({match.sum()} matched times)')
    print(f'   max |diff| x, y, z = {np.abs(diff).max(axis=0)}  (scale {scale:.3e} N)')
    print(f'   max relative diff (x, y) = {np.abs(diff[:, :2]).max() / scale:.2e}')
    print(f'   last step: sum(Fs) = {Fsum[-1]},  LiftandDrag = {L[idx[-1]]}')

    # 2. z-force
    print(f'\n2. max |sum Fz| / max |sum Fx| = '
          f'{np.abs(Fsum[:, 2]).max() / np.abs(Fsum[:, 0]).max():.2e},  '
          f'max nodal |Fz| = {np.abs(Fs[..., 2]).max():.2e} N')

    # 3. effective span from the stagnation nodes (front of the cylinder, min x)
    fs = newRun.freestream(mach, Re)
    q = 0.5 * fs['rho'] * fs['U'] ** 2
    g = newRun.gamma
    p02_p1 = (((g + 1) ** 2 * mach ** 2) / (4 * g * mach ** 2 - 2 * (g - 1))) ** (g / (g - 1)) \
        * (1 - g + 2 * g * mach ** 2) / (g + 1)          # Rayleigh pitot formula
    cpMax = (p02_p1 - 1) / (0.5 * g * mach ** 2)
    xc, yc = X[:, 0].mean(), X[:, 1].mean()
    theta = np.arctan2(X[:, 1] - yc, X[:, 0] - xc)
    dFront = np.abs(np.abs(theta) - math.pi)                 # angle from the front point
    front = dFront < dFront.min() + 1e-9                      # nearest ring position, all z
    nTheta = len(np.unique(np.round(theta, 8)))
    arc = math.pi * mesh.D / nTheta
    tAvg = tF > 0.5 * tF[-1]                          # average over the second half
    FxFront = Fs[tAvg][:, front, 0].sum(axis=1).mean()
    dz = mesh.dz
    zLen = X[:, 2].max() - X[:, 2].min()
    print(f'\n3. effective span (stagnation nodes: {front.sum()}, arc per node {arc:.4f} m)')
    print(f'   Cp_max (normal shock pitot) = {cpMax:.3f},  p0 - p_inf = {cpMax * q:.4f} Pa')
    print(f'   sum Fx on stagnation nodes (2nd-half average) = {FxFront:.4e} N')
    print(f'   -> span = Fx / ((p0 - p_inf) * arc) = {FxFront / (cpMax * q * arc):.3f} m')
    print(f'   slab thickness dz = {dz:.3f} m,  tube length = {zLen:.3f} m')

    # Drag coefficient with either span, for reference
    Fx = Fsum[tAvg, 0].mean()
    print(f'\n   Cd (2nd-half average) with span = dz: {Fx / (q * mesh.D * dz):.3f},  '
          f'with span = tube length: {Fx / (q * mesh.D * zLen):.3f}')
