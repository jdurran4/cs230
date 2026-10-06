"""
Phase 4, multi-grid binning: bin each converged solution onto MANY coarse grids.

The coarse control volumes are a post-processing choice: AERO-F never sees them. So one
fine solution can be filtered onto coarse grids of any spacing h and origin, like an a
priori LES study filtering one DNS at several filter widths. Per run this gives
  grid 0       the simulation's own coarse grid (h = D / 1.2), which reproduces
               binInputs + binTargets + buildDataset (check with  binMulti.py check)
  grids 1..K   D/h log-uniform in DhRange and the cylinder center at (gox, goy) h from a
               grid point, gox, goy uniform in [0, 1); a Latin hypercube over the K grids
               (seeded by the run name, so reruns give the same grids)

Coarse grid: nodes (xo + i h, yo + j h, {0, dz}), each hex split into the 6 tets of
cylMeshGen.fluidTets, median-dual columns. Those 6 tets are the Kuhn simplices along the
c000-c111 diagonal, which gives the owning column of a point in closed form (dualColumn),
with the same rule as binOneCell.owningNode: the vertex with the largest barycentric
coordinate. Every interior column has volume h^2 dz.

Inputs are fluid-volume averages of the fine solution's conserved variables over each
column (fine tet pieces V/4 per vertex, placed halfway between vertex and centroid;
pieces whose vertex is inside the body are dropped), as in binInputs.py. Targets are
the per-station surface forces split among columns by z-sampling the slab, as in
binTargets.py. Only the force columns (the ones the surface passes through) are kept,
at the final (converged) step.

Rows dropped: columns that may stick out of the fluid domain (grid point closer than h
to the domain boundary; a median-dual column extends less than h from its grid point;
only happens for the largest h), columns with no fluid piece (NaN averages), and
columns the body fills completely. cover (fine-piece volume / h^2 dz) is kept as a
diagnostic: it scatters by about 1% because pieces straddle column boundaries.

Columns: the same as buildDataset.py (so ablation.py runs on either), plus
  grid (0..K), gox, goy (cylinder center offset from its grid point, in h), cover (fine
  volume / h^2 dz).  h, Dh, dh, rpar, rperp use the grid's h.

Usage (from aerof/):
  ../.venv/bin/python binMulti.py RUN [RUN ...]     bin these runs -> runs/RUN/samples.npz
  ../.venv/bin/python binMulti.py                   every finished run not binned yet
  ../.venv/bin/python binMulti.py check RUN         grid 0 vs dataset.npz for an old run
  ../.venv/bin/python binMulti.py merge             runs/*/samples.npz -> dataset_multi.npz
  4 at a time:  ls runs | xargs -P 4 -n 1 ../.venv/bin/python binMulti.py
"""
import os
import re
import sys
import zlib

import numpy as np

import binInputs as bi
import binOneCell as bin1
import binTargets as bt
import checkSurfaceForce as chk
import cylMeshGen as mesh
import newRun

# ============= User-defined parameters ==============================================
nGrids = 20                # random grids per run (plus grid 0, the simulation's own)
DhRange = (0.25, 2.4)      # D/h, sampled uniformly in log
gridSeed = 0
nDisk = 400                # disk samples per diameter for the solid volume
nzDisk = 16                # z samples for the solid volume
nzSample = 64              # z samples per surface station for the force split

aerofDir = os.path.dirname(os.path.abspath(__file__))
runsDir = os.path.join(aerofDir, 'runs')
g, R_gas = newRun.gamma, newRun.R_gas
nameRe = re.compile(r'M([\d.]+)_Re([\d.]+)_x([\d.]+)_y([\d.]+)(?:_(.*))?$')


# ============= Coarse grid ==========================================================
def dualColumn(P, xo, yo, h, dz):
    """Grid indices (i, j) of the median-dual column containing each point of P (n, 3).
    In local hex coordinates s in [0, 1)^3 the containing Kuhn tet is the path
    c000 -> +e_a -> +e_a+e_b -> c111, where a, b, c sort s in descending order; its
    barycentric coordinates are (1 - s_a, s_a - s_b, s_b - s_c, s_c). The owning vertex
    k (0..3) is the largest; it sits at c000 plus the first k unit steps of the path."""
    fx, fy = (P[:, 0] - xo) / h, (P[:, 1] - yo) / h
    i0, j0 = np.floor(fx), np.floor(fy)
    s = np.column_stack([fx - i0, fy - j0, np.clip(P[:, 2] / dz, 0, 1)])
    order = np.argsort(-s, axis=1)                                    # path axes a, b, c
    ss = np.take_along_axis(s, order, axis=1)
    lam = np.column_stack([1 - ss[:, 0], ss[:, 0] - ss[:, 1], ss[:, 1] - ss[:, 2], ss[:, 2]])
    k = lam.argmax(axis=1)
    taken = np.arange(3)[None, :] < k[:, None]                        # steps 0..k-1 done
    di = ((order == 0) & taken).sum(axis=1)
    dj = ((order == 1) & taken).sum(axis=1)
    return (i0 + di).astype(np.int64), (j0 + dj).astype(np.int64)


def gridList(runName, geo):
    """(Dh, gox, goy, xo, yo) of grid 0 (the simulation's) and the nGrids random ones."""
    D, h0, xc, yc = geo['D'], geo['h'], geo['xc'], geo['yc']
    xo0, yo0 = mesh.x0 - mesh.Lx / 2, mesh.y0 - mesh.Ly / 2
    gox0, goy0 = ((xc - xo0) / h0) % 1, ((yc - yo0) / h0) % 1
    grids = [(D / h0, gox0, goy0, xc - gox0 * h0, yc - goy0 * h0)]
    rng = np.random.default_rng([gridSeed, zlib.crc32(runName.encode())])
    u = (np.array([rng.permutation(nGrids) for _ in range(3)]).T
         + rng.random((nGrids, 3))) / nGrids                          # Latin hypercube
    for uDh, ux, uy in u:
        Dh = np.exp(np.log(DhRange[0]) + uDh * np.log(DhRange[1] / DhRange[0]))
        h = D / Dh
        grids.append((Dh, ux, uy, xc - ux * h, yc - uy * h))
    return grids


# ============= Per run: read once ===================================================
def readRun(runName):
    """Everything about the converged solution that does not depend on the grid."""
    runDir = os.path.join(runsDir, runName)
    post = os.path.join(runDir, 'postpro')
    geo = mesh.loadGeometry(runDir)
    mach, rhoInf, pInf = bt.readInlet(os.path.join(runDir, 'FluidFile'))
    U = mach * np.sqrt(g * pInf / rhoInf)

    # fine mesh and fields of the last output frame, last snapshot
    fr = sorted(f.split('.')[-1] for f in os.listdir(post)
                if f.startswith('OUTPUT.top.') and not f.endswith('.dec'))[-1]
    fineNodes, fineTets = bin1.readDomain(os.path.join(post, f'OUTPUT.top.{fr}'))
    t, rho = bi.readXpost(os.path.join(post, f'Density.{fr}.xpost'))
    _, vel = bi.readXpost(os.path.join(post, f'Velocity.{fr}.xpost'))
    _, p = bi.readXpost(os.path.join(post, f'Pressure.{fr}.xpost'))
    rho, vel, p = rho[-1], vel[-1], p[-1]
    rhoE = p / (g - 1) + 0.5 * rho * (vel ** 2).sum(axis=1)

    # fine tet pieces (binInputs.frameWeights): V/4 per vertex, halfway to the centroid
    V = bi.tetVolumes(fineNodes, fineTets)
    cen = fineNodes[fineTets].mean(axis=1)
    pts = (0.5 * (fineNodes[fineTets] + cen[:, None, :])).reshape(-1, 3)
    node = fineTets.ravel()
    w = np.repeat(V / 4, 4)
    fluid = ~bi.insideBody(fineNodes[node, :2], geo)
    cons = np.column_stack([rho, rho[:, None] * vel, rhoE])[node]    # rho, rho u (3), rho E

    # solid samples (binInputs.solidVolume): the disk x [0, dz]
    D, dz = geo['D'], geo['dz']
    s = (np.arange(nDisk) + 0.5) / nDisk - 0.5
    X, Y = np.meshgrid(geo['xc'] + D * s, geo['yc'] + D * s)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    xy = xy[bi.insideBody(xy, geo)]
    z = (np.arange(nzDisk) + 0.5) / nzDisk * dz
    solid = np.column_stack([np.tile(xy, (nzDisk, 1)), np.repeat(z, len(xy))])
    dVsolid = (D / nDisk) ** 2 * dz / nzDisk

    # surface force per x-y station at the last step (binTargets)
    X = chk.readSurfaceNodes(geo['surfFile'])
    tF, Fs = chk.readSurfaceForce(os.path.join(post, 'EmbeddedSurfaceForce'))
    assert abs(tF[-1] - t[-1]) <= 1e-5 * abs(t[-1]), 'force and field times differ'
    sxy, station = np.unique(np.round(X[:, :2], 12), axis=0, return_inverse=True)
    Fst = np.zeros((len(sxy), 2))
    np.add.at(Fst, station.ravel(), Fs[-1, :, :2])
    zs = (np.arange(nzSample) + 0.5) / nzSample * dz
    surf = np.column_stack([np.repeat(sxy, nzSample, axis=0), np.tile(zs, len(sxy))])

    return dict(geo=geo, mach=mach, rhoInf=rhoInf, pInf=pInf, q=0.5 * rhoInf * U ** 2,
                pts=pts, w=w, fluid=fluid, cons=cons, solid=solid, dVsolid=dVsolid,
                sxy=sxy, Fst=Fst, nds=bt.stationNormals(sxy), surf=surf)


# ============= Per grid =============================================================
def binGrid(R, Dh, gox, goy, xo, yo):
    """Rows (dict of arrays) for the force columns of one coarse grid."""
    geo = R['geo']
    D, dz, span = geo['D'], geo['dz'], geo['span']
    h = D / Dh
    key = lambda ij: (ij[0] + 10000) * 100000 + (ij[1] + 10000)     # (i, j) -> one int

    # force columns and the force split (frac: station x column)
    ks = key(dualColumn(R['surf'], xo, yo, h, dz))
    cols, sc = np.unique(ks, return_inverse=True)
    ns = len(R['sxy'])
    frac = np.zeros((ns, len(cols)))
    np.add.at(frac, (np.repeat(np.arange(ns), nzSample), sc), 1.0 / nzSample)
    colXY = np.column_stack([xo + (cols // 100000 - 10000) * h, yo + (cols % 100000 - 10000) * h])

    # fine pieces -> force columns
    kp = key(dualColumn(R['pts'], xo, yo, h, dz))
    pos = np.searchsorted(cols, kp)
    hit = (pos < len(cols)) & (cols[np.minimum(pos, len(cols) - 1)] == kp)
    pc, w, fl = pos[hit], R['w'][hit], R['fluid'][hit]
    nc = len(cols)
    cover = np.bincount(pc, weights=w, minlength=nc) / (h * h * dz)
    Wf = np.bincount(pc[fl], weights=w[fl], minlength=nc)
    avg = np.column_stack([np.bincount(pc[fl], weights=w[fl] * R['cons'][hit][fl, m],
                                       minlength=nc) for m in range(5)])
    with np.errstate(invalid='ignore', divide='ignore'):
        avg /= Wf[:, None]

    # solid volume and body-piece centroid per force column
    kd = key(dualColumn(R['solid'], xo, yo, h, dz))
    pos = np.searchsorted(cols, kd)
    hit = (pos < len(cols)) & (cols[np.minimum(pos, len(cols) - 1)] == kd)
    nsol = np.bincount(pos[hit], minlength=nc)
    solidCen = np.column_stack([np.bincount(pos[hit], weights=R['solid'][hit, d], minlength=nc)
                                for d in range(2)]) / np.maximum(nsol, 1)[:, None]
    volFrac = nsol * R['dVsolid'] / (h * h * dz)

    # derived from the averaged conserved state (binInputs)
    rho, rhou, rhoE = avg[:, 0], avg[:, 1:4], avg[:, 4]
    u = rhou[:, :2] / rho[:, None]
    p = (g - 1) * (rhoE - 0.5 * (rhou ** 2).sum(axis=1) / rho)
    T = p / (rho * R_gas)
    speed = np.linalg.norm(u, axis=1)
    M = speed / np.sqrt(g * p / rho)
    mu = newRun.C_suth * T ** 1.5 / (T + newRun.S_suth)
    ReD = rho * speed * D / mu

    # targets (binTargets) and features (buildDataset)
    nds = R['nds']
    S = -(frac.T @ R['Fst'] - R['pInf'] * span * (frac.T @ nds)) / span
    N = frac.T @ nds / D
    e = u / speed[:, None]
    ePerp = np.column_stack([-e[:, 1], e[:, 0]])
    pBar, qBar = p, 0.5 * rho * speed ** 2
    Sp = S - pBar[:, None] * D * N
    r = solidCen - colXY
    r[nsol == 0] = np.nan
    rn = np.linalg.norm(r, axis=1)
    dot = lambda a, b: (a * b).sum(axis=1)
    q = R['q']
    full = lambda v: np.full(nc, v, dtype=float)
    return dict(
        colX=colXY[:, 0], colY=colXY[:, 1], h=full(h), cover=cover,
        gox=full(gox), goy=full(goy),
        Mbar=M, ReD=ReD, volFrac=volFrac,
        Npar=dot(N, e), Nperp=dot(N, ePerp), Nmag=np.linalg.norm(N, axis=1),
        cosT=dot(r, e) / rn, sinT=dot(r, ePerp) / rn, dh=rn / h,
        rpar=dot(r, e) / h, rperp=dot(r, ePerp) / h, Dh=full(Dh),
        pqRatio=pBar / qBar,
        SxInf=S[:, 0] / (q * D), SyInf=S[:, 1] / (q * D),
        Spar_q=dot(Sp, e) / (qBar * D), Sperp_q=dot(Sp, ePerp) / (qBar * D),
        Spar_pq=dot(Sp, e) / ((pBar + qBar) * D), Sperp_pq=dot(Sp, ePerp) / ((pBar + qBar) * D),
        SFpar_q=dot(S, e) / (qBar * D), SFperp_q=dot(S, ePerp) / (qBar * D),
        SFpar_pq=dot(S, e) / ((pBar + qBar) * D), SFperp_pq=dot(S, ePerp) / ((pBar + qBar) * D),
        Sx=S[:, 0], Sy=S[:, 1], pBar=pBar, qBar=qBar, ePar_x=e[:, 0], ePar_y=e[:, 1])


def binRun(runName, grids=None):
    R = readRun(runName)
    geo = R['geo']
    m = nameRe.match(runName)
    tag = m[5] or ''
    parts = []
    for k, grid in enumerate(grids or gridList(runName, geo)):
        rows = binGrid(R, *grid)
        n = len(rows['colX'])
        full = lambda v: np.full(n, v, dtype=float)
        rows.update(grid=full(k), machInf=full(float(m[1])), ReInf=full(float(m[2])),
                    ox=full(float(m[3])), oy=full(float(m[4])), zOverhang=full(geo['zOverhang']),
                    xc=full(geo['xc']), yc=full(geo['yc']), D=full(geo['D']), qInf=full(R['q']),
                    duplicate=full(tag in ('me2M', 'zo0.1')))
        parts.append(rows)
    rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    sumS = [rows['SxInf'][rows['grid'] == k].sum() for k in range(len(parts))]
    inDomain = ((np.abs(rows['colX'] - mesh.x0) <= mesh.Lx / 2 - rows['h'])
                & (np.abs(rows['colY'] - mesh.y0) <= mesh.Ly / 2 - rows['h']))
    keep = inDomain & np.all(
        [np.isfinite(rows[k]) for k in ('Mbar', 'ReD', 'Spar_pq', 'Sperp_pq')], axis=0)
    keep &= rows['volFrac'] < 1                    # the body fills the whole column: no flow
    print(f'{runName}: {len(parts)} grids, {keep.sum()} of {len(keep)} rows kept '
          f'({np.sum(~inDomain)} near the domain boundary); '
          f'-Cd over grids {-max(sumS):.4f} .. {-min(sumS):.4f} (must agree)', flush=True)
    return {k: v[keep] for k, v in rows.items()}


# ============= Main =================================================================
def finishedRuns():
    return sorted(r for r in os.listdir(runsDir)
                  if os.path.isfile(os.path.join(runsDir, r, 'DONE'))
                  and os.path.isfile(os.path.join(runsDir, r, 'postpro', 'EmbeddedSurfaceForce')))


def merge():
    runs = [r for r in finishedRuns()
            if os.path.isfile(os.path.join(runsDir, r, 'samples.npz'))]
    parts = []
    for k, r in enumerate(runs):
        d = dict(np.load(os.path.join(runsDir, r, 'samples.npz')))
        d['run'] = np.full(len(d['colX']), float(k))
        parts.append(d)
    keys = list(parts[0])
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out = os.path.join(aerofDir, 'dataset_multi.npz')
    np.savez(out, runNames=np.array(runs), featureNames=np.array(keys), **data)
    print(f'{len(runs)} runs, {len(data["run"])} rows -> {out}')


def check(runName):
    """Grid 0 must reproduce the single-grid pipeline (dataset.npz) for an old run."""
    old = np.load(os.path.join(aerofDir, 'dataset.npz'))
    k = list(old['runNames']).index(runName)
    sel = old['run'] == k
    R = readRun(runName)
    new = binGrid(R, *gridList(runName, R['geo'])[0])
    oi = np.lexsort((old['colY'][sel], old['colX'][sel]))
    ni = np.lexsort((new['colY'], new['colX']))
    assert np.allclose(old['colX'][sel][oi], new['colX'][ni]) and \
        np.allclose(old['colY'][sel][oi], new['colY'][ni]), 'different force columns'
    for f in ('Mbar', 'ReD', 'volFrac', 'Npar', 'Nperp', 'dh', 'sinT', 'SxInf', 'SyInf',
              'Spar_pq', 'Sperp_pq'):
        print(f'{f:>9}: max |new - old| = {np.nanmax(np.abs(new[f][ni] - old[f][sel][oi])):.2e}'
              f'   (max |old| {np.nanmax(np.abs(old[f][sel])):.3g})')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['merge']:
        merge()
    elif args[:1] == ['check']:
        check(args[1])
    else:
        for r in args or finishedRuns():
            out = os.path.join(runsDir, r, 'samples.npz')
            if not args and os.path.isfile(out):
                continue
            np.savez(out, **binRun(r))
