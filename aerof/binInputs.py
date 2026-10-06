"""
Phase 4 inputs: fluid state averaged over each coarse control volume, per output step.

The model will see the coarse solver's cell-averaged CONSERVED variables online, so we
volume-average rho, rho u, rho E over the FLUID part of each coarse median-dual column
and derive everything else (u, p, T, Mach, Re_D, flow angle) from those averages.

Fine-mesh volumes: each output frame NNNN has its own mesh (postpro/OUTPUT.top.NNNN) and
fields (postpro/<QOI>.NNNN.xpost). Frame 0000 is output on the mesh the run started from
(the coarse mesh); later frames
follow refinements. Each fine tet gives V/4 to each of its vertices
(median dual). That piece is assigned to a coarse column at the point halfway between
the vertex and the tet centroid (inside the piece, so no ties on coarse dual
boundaries; exact wherever the fine mesh is the coarse mesh). Pieces whose vertex is
inside the body are dropped: AERO-F's values there are ghost states, not flow.

Volume fraction is geometric: the body cross-section (disk) is sampled in x-y and z and
assigned to columns with the same median-dual rule. The cylinder center is per case
(binTargets.geometry). The fine-mesh fluid volume is a
cross-check, not used.

Writes runs/<run>/inputs.npz with, for ALL coarse columns (ncol) and output times (nt):
  times (nt,), columns (ncol, 2) grid-point x-y
  rho (nt, ncol), rhou (nt, ncol, 3), rhoE (nt, ncol)       averaged conserved variables
  u (nt, ncol, 2), p, T, M, ReD, flowAngle (nt, ncol)       derived from the averages
  fluidCentroid (nt, ncol, 2)                               x-y centroid of the fluid part
  Vcell, Vsolid, volFrac (ncol,), VfluidFine (nt, ncol)     geometry (m^3)
  solidCentroid (ncol, 2)                                   x-y centroid of the body piece
  xc, yc, D                                                 cylinder center, diameter

Usage: set runNames below (None = every run with postpro/*.NNNN.xpost), then run.
"""
import glob
import os

import numpy as np

import binOneCell as bin1
import binTargets
import cylMeshGen as mesh
import newRun

# ============= User-defined parameters ==============================================
runNames = None
nDisk = 400                # disk samples per diameter for the solid volume
nzDisk = 16                # z samples for the solid volume

aerofDir = os.path.dirname(os.path.abspath(__file__))
runsDir = os.path.join(aerofDir, 'runs')
g, R_gas = newRun.gamma, newRun.R_gas


# ============= Readers ==============================================================
def readXpost(path):
    """Returns times (nt,) and values (nt, nn) or (nt, nn, 3)."""
    with open(path) as f:
        kind = f.readline().split()[0]                    # Scalar or Vector
        nn = int(f.readline())
        data = np.fromstring(f.read(), sep=' ')
    nc = 3 if kind == 'Vector' else 1
    per = 1 + nn * nc
    nt = len(data) // per                                 # drops a partial last step
    data = data[:nt * per].reshape(nt, per)
    vals = data[:, 1:].reshape(nt, nn, nc)
    return data[:, 0], vals[..., 0] if nc == 1 else vals


# ============= Geometry =============================================================
def coarseColumns(nodes, tets):
    """Column id of every coarse node, column x-y, and column volume (V/4 per vertex)."""
    columns, colOf = np.unique(np.round(nodes[:, :2], 9), axis=0, return_inverse=True)
    V = tetVolumes(nodes, tets)
    Vcell = np.zeros(len(columns))
    np.add.at(Vcell, colOf.ravel()[tets], np.repeat(V[:, None] / 4, 4, axis=1))
    return colOf.ravel(), columns, Vcell


def tetVolumes(nodes, tets):
    v = nodes[tets]
    return np.abs(np.einsum('ij,ij->i', v[:, 1] - v[:, 0],
                            np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]))) / 6


def insideBody(xy, geo):
    return np.hypot(xy[:, 0] - geo['xc'], xy[:, 1] - geo['yc']) < geo['D'] / 2


def solidVolume(nodes, tets, colOf, ncol, geo):
    """Body volume inside each column, by sampling the disk x [0, dz], and the x-y
    centroid of that body piece (0 where the column holds no body)."""
    D, dz = geo['D'], geo['dz']
    s = (np.arange(nDisk) + 0.5) / nDisk - 0.5
    X, Y = np.meshgrid(geo['xc'] + D * s, geo['yc'] + D * s)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    xy = xy[insideBody(xy, geo)]
    z = (np.arange(nzDisk) + 0.5) / nzDisk * dz
    P = np.column_stack([np.tile(xy, (nzDisk, 1)), np.repeat(z, len(xy))])
    col = colOf[bin1.owningNode(P, nodes, tets)]
    dV = (D / nDisk) ** 2 * dz / nzDisk
    n = np.bincount(col, minlength=ncol)
    centroid = np.column_stack([np.bincount(col, weights=P[:, d], minlength=ncol)
                                for d in range(2)]) / np.maximum(n, 1)[:, None]
    return n * dV, centroid


def frameWeights(fineNodes, fineTets, nodes, tets, colOf, ncol, geo):
    """Sparse averaging operator for one AMR frame: (col, fineNode, volume) triples
    over fluid pieces, and the x-y centroid of each column's fluid pieces."""
    V = tetVolumes(fineNodes, fineTets)
    cen = fineNodes[fineTets].mean(axis=1)
    pts = 0.5 * (fineNodes[fineTets] + cen[:, None, :])                     # (ne, 4, 3)
    node = fineTets.ravel()
    w = np.repeat(V / 4, 4)
    col = colOf[bin1.owningNode(pts.reshape(-1, 3), nodes, tets)]
    fluid = ~insideBody(fineNodes[node, :2], geo)
    Vtot = w.sum()

    col, node, w, xy = col[fluid], node[fluid], w[fluid], pts.reshape(-1, 3)[fluid, :2]
    key, inv = np.unique(col * len(fineNodes) + node, return_inverse=True)
    W = np.bincount(inv.ravel(), weights=w)
    Wcol = np.bincount(col, weights=w, minlength=ncol)
    centroid = np.column_stack([np.bincount(col, weights=w * xy[:, d], minlength=ncol)
                                for d in range(2)]) / np.maximum(Wcol, 1e-300)[:, None]
    return (key // len(fineNodes), key % len(fineNodes), W), Wcol, centroid, Vtot


def average(op, Wcol, q):
    """Fluid-volume average per column of nodal values q (nt, nn) -> (nt, ncol)."""
    col, node, W = op
    out = np.zeros((q.shape[0], len(Wcol)))
    for t in range(q.shape[0]):
        out[t] = np.bincount(col, weights=W * q[t, node], minlength=len(Wcol))
    return out / np.where(Wcol > 0, Wcol, np.nan)


# ============= Main per run =========================================================
def binInputs(runName, nodes, tets):
    runDir = binTargets.caseDir(runName)
    post = os.path.join(runDir, 'postpro')
    geo = binTargets.geometry(runName)
    colOf, columns, Vcell = coarseColumns(nodes, tets)
    ncol = len(columns)
    Vsolid, solidCentroid = solidVolume(nodes, tets, colOf, ncol, geo)

    frames = sorted(f.split('.')[-1] for f in glob.glob(os.path.join(post, 'OUTPUT.top.[0-9]*'))
                    if not f.endswith('.dec'))
    acc = {k: [] for k in ('times', 'rho', 'rhou', 'rhoE', 'cen', 'Vf')}
    for fr in frames:
        fineNodes, fineTets = bin1.readDomain(os.path.join(post, f'OUTPUT.top.{fr}'))
        op, Wcol, centroid, Vtot = frameWeights(fineNodes, fineTets, nodes, tets, colOf, ncol,
                                                geo)
        t, rho = readXpost(os.path.join(post, f'Density.{fr}.xpost'))
        _, vel = readXpost(os.path.join(post, f'Velocity.{fr}.xpost'))
        _, p = readXpost(os.path.join(post, f'Pressure.{fr}.xpost'))
        nt = min(len(t), len(vel), len(p))
        t, rho, vel, p = t[:nt], rho[:nt], vel[:nt], p[:nt]
        rhoE = p / (g - 1) + 0.5 * rho * (vel ** 2).sum(axis=2)
        acc['times'].append(t)
        acc['rho'].append(average(op, Wcol, rho))
        acc['rhou'].append(np.stack([average(op, Wcol, rho * vel[..., d]) for d in range(3)],
                                    axis=2))
        acc['rhoE'].append(average(op, Wcol, rhoE))
        acc['cen'].append(np.repeat(centroid[None], nt, axis=0))
        acc['Vf'].append(np.repeat(Wcol[None], nt, axis=0))
        print(f'{runName} frame {fr}: {len(fineNodes)} fine nodes, {nt} snapshots, '
              f'fine volume {Vtot:.6f} vs coarse {Vcell.sum():.6f} m^3')
    times = np.concatenate(acc['times'])
    rho, rhou, rhoE = (np.concatenate(acc[k]) for k in ('rho', 'rhou', 'rhoE'))

    # derived from the averaged conserved state
    u = rhou[..., :2] / rho[..., None]
    speed = np.linalg.norm(u, axis=2)
    p = (g - 1) * (rhoE - 0.5 * (rhou ** 2).sum(axis=2) / rho)
    T = p / (rho * R_gas)
    M = speed / np.sqrt(g * p / rho)
    mu = newRun.C_suth * T ** 1.5 / (T + newRun.S_suth)
    ReD = rho * speed * geo['D'] / mu
    flowAngle = np.arctan2(u[..., 1], u[..., 0])

    out = os.path.join(runDir, 'inputs.npz')
    np.savez(out, times=times, columns=columns, rho=rho, rhou=rhou, rhoE=rhoE, u=u, p=p,
             T=T, M=M, ReD=ReD, flowAngle=flowAngle, fluidCentroid=np.concatenate(acc['cen']),
             Vcell=Vcell, Vsolid=Vsolid, volFrac=Vsolid / Vcell, solidCentroid=solidCentroid,
             VfluidFine=np.concatenate(acc['Vf']), xc=geo['xc'], yc=geo['yc'], D=geo['D'])

    # checks / summary
    mach, rhoInf, pInf = binTargets.readInlet(os.path.join(runDir, 'FluidFile'))
    far = np.argmin(np.linalg.norm(columns - [mesh.x0 - mesh.Lx / 2 + mesh.h, 0.0], axis=1))
    print(f'   solid: sum Vsolid = {Vsolid.sum():.5f}, pi D^2/4 dz = '
          f'{np.pi * geo["D"] ** 2 / 4 * geo["dz"]:.5f} m^3')
    print(f'   inflow column ({columns[far, 0]:.2f}, {columns[far, 1]:.2f}) at last step: '
          f'rho/rho_inf = {rho[-1, far] / rhoInf:.5f}, p/p_inf = {p[-1, far] / pInf:.5f}, '
          f'M = {M[-1, far]:.4f} (inlet {mach:g}), Re_D = {ReD[-1, far]:.2f}')
    tAvg = times > 0.5 * times[-1]
    hit = np.flatnonzero(Vsolid > 0)
    print(f'   columns cut by the body, 2nd-half average:')
    print(f'   {"grid point x, y":>18}  {"volFrac":>7}  {"Vfluid fine/geom":>16}  {"M":>6}  '
          f'{"Re_D":>6}  {"angle":>6}  {"p/pinf":>6}  {"rho/rhoinf":>10}')
    for c in hit[np.argsort(-Vsolid[hit])]:
        print(f'   {columns[c, 0]:8.3f} {columns[c, 1]:8.3f}  {Vsolid[c] / Vcell[c]:7.3f}  '
              f'{acc["Vf"][-1][0, c] / (Vcell[c] - Vsolid[c]):16.4f}  '
              f'{M[tAvg, c].mean():6.3f}  {ReD[tAvg, c].mean():6.1f}  '
              f'{np.degrees(flowAngle[tAvg, c].mean()):6.1f}  {p[tAvg, c].mean() / pInf:6.3f}  '
              f'{rho[tAvg, c].mean() / rhoInf:10.3f}')
    print(f'   wrote {out}')


if __name__ == '__main__':
    names = runNames or sorted(
        r for r in os.listdir(runsDir)
        if glob.glob(os.path.join(binTargets.caseDir(r), 'postpro', '*.[0-9][0-9][0-9][0-9].xpost')))
    nodes, tets = bin1.readDomain(bin1.domainFile)
    for name in names:
        binInputs(name, nodes, tets)
