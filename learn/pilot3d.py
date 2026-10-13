"""Pilot: bin one resolved run onto a coarse mesh extruded to several z-layers, with
true 3D median-dual cells (one per node), and compare with the quasi-2D column targets.

The resolved flow is z-uniform, so any coarse mesh thickness can be used: a surface
station at (x, y) carries force per unit span f_s, and the force on the cable piece
inside 3D cell K is  F_K = sum_s f_s L_K(x_s, y_s), where L_K is the length of the
vertical line through the station that lies in K. L_K comes from z-sampling the
extruded mesh and assigning each sample to the vertex with the largest barycentric
coordinate of its containing tet (the median-dual rule), found by a generic tet search.

Coarse mesh: the x-y grid of steady-v0/sources/domain.top (h = D/1.2), extruded to
nLayers layers of thickness hz, each hex split into the same 6 Kuhn tets as cylMeshGen.
Interior cells (levels 1..nLayers-1) are the training cells; the symmetry-plane levels
0 and nLayers are printed only for comparison.

Checks, for run M1.5_Re100_x0.04_y0.12:
  1. sum over all 3D cells of F == sum over stations (nothing lost)
  2. interior cells, per unit span (F_K / hz): the same at every interior level, and
     equal to the old column target S of targets.npz
  3. volume fraction of interior cells vs the old inputs.npz volFrac

Usage (from the repo root):  python -m learn.pilot3d
"""
import json
import os

import numpy as np

import common

# ============= User-defined parameters ==============================================
runName = 'M1.5_Re100_x0.04_y0.12'
nLayers = 4                # coarse layers in z
nzSample = 64              # z samples per layer
nDisk = 400                # disk samples per diameter for the solid volume

runDir = os.path.join(common.STEADY, 'runs', runName)
domainFile = os.path.join(common.STEADY, 'sources', 'domain.top')


# ============= Coarse 3D mesh =======================================================
def extrudedMesh(xs, ys, hz, nz):
    """Nodes (nx ny (nz+1), 3), x fastest, then y, then z; 6 Kuhn tets per hex along the
    c000-c111 diagonal (the cylMeshGen split, repeated in every layer)."""
    nx, ny = len(xs), len(ys)
    Z, Y, X = np.meshgrid(np.arange(nz + 1) * hz, ys, xs, indexing='ij')
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    k, j, i = (a.ravel() for a in np.meshgrid(np.arange(nz), np.arange(ny - 1),
                                              np.arange(nx - 1), indexing='ij'))
    nid = lambda di, dj, dk: (i + di) + nx * (j + dj) + nx * ny * (k + dk)
    c = {f'{a}{b}{d}': nid(a, b, d) for a in (0, 1) for b in (0, 1) for d in (0, 1)}
    split = (('000', '100', '110', '111'), ('000', '100', '101', '111'),
             ('000', '010', '110', '111'), ('000', '010', '011', '111'),
             ('000', '001', '011', '111'), ('000', '001', '101', '111'))
    tets = np.concatenate([np.column_stack([c[v] for v in t]) for t in split])
    return nodes, tets


def owningNode(P, nodes, tets, chunk=4000):
    """Coarse node whose median-dual cell contains each point of P (np, 3): find the
    containing tet, then take the vertex with the largest barycentric coordinate."""
    Vall = nodes[tets]                                                      # (ne, 4, 3)
    Minv = np.linalg.inv((Vall[:, 1:] - Vall[:, :1]).transpose(0, 2, 1))    # (ne, 3, 3)
    lo, hi = Vall.min(axis=1), Vall.max(axis=1)
    order = np.lexsort((P[:, 2], P[:, 1], P[:, 0]))
    vtx = np.empty(len(P), dtype=int)
    for s in range(0, len(P), chunk):
        idx = order[s:s + chunk]
        p = P[idx]
        near = np.all((hi >= p.min(axis=0) - 1e-9) & (lo <= p.max(axis=0) + 1e-9), axis=1)
        T, V, M = tets[near], Vall[near], Minv[near]
        lam123 = np.einsum('kij,pkj->pki', M, p[:, None, :] - V[None, :, 0])
        lam = np.concatenate([1 - lam123.sum(axis=2, keepdims=True), lam123], axis=2)
        inside = np.all(lam > -1e-12, axis=2)
        if not inside.any(axis=1).all():
            raise RuntimeError('some sample points are in no coarse tet')
        k = inside.argmax(axis=1)
        vtx[idx] = T[k, lam[np.arange(len(p)), k].argmax(axis=1)]
    return vtx


def lengthsInCells(xy, nodes, tets, thickness, nz):
    """L (np, nn): length of the vertical line at each x-y point inside each node's cell."""
    z = (np.arange(nz) + 0.5) / nz * thickness
    P = np.column_stack([np.repeat(xy, nz, axis=0), np.tile(z, len(xy))])
    vtx = owningNode(P, nodes, tets)
    L = np.zeros((len(xy), len(nodes)))
    np.add.at(L, (np.repeat(np.arange(len(xy)), nz), vtx), thickness / nz)
    return L


def dualVolumes(nodes, tets):
    v = nodes[tets]
    V = np.abs(np.einsum('ij,ij->i', v[:, 1] - v[:, 0],
                         np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]))) / 6
    Vn = np.zeros(len(nodes))
    np.add.at(Vn, tets, np.repeat(V[:, None] / 4, 4, axis=1))
    return Vn


# ============= Resolved run: station forces per unit span ===========================
def readSurfaceForce(path):
    """Times (nt,) and forces (nt, nn, 3) of an EmbeddedSurfaceForce file."""
    with open(path) as f:
        f.readline()
        nn = int(f.readline())
        lines = f.read().split('\n')
    nt = len(lines) // (nn + 1)
    steps = [lines[k * (nn + 1):(k + 1) * (nn + 1)] for k in range(nt)]
    return (np.array([float(s[0]) for s in steps]),
            np.array([[r.split() for r in s[1:]] for s in steps], dtype=float))


def stationNormals(xy):
    """Tent-weighted outward normal x arc length of each station (per unit span)."""
    cen = xy.mean(axis=0)
    order = np.argsort(np.arctan2(xy[:, 1] - cen[1], xy[:, 0] - cen[0]))
    seg = np.roll(xy[order], -1, axis=0) - xy[order]
    segN = np.column_stack([seg[:, 1], -seg[:, 0]])
    nds = np.empty_like(xy)
    nds[order] = 0.5 * (segN + np.roll(segN, 1, axis=0))
    return nds


def stationSources(runDir):
    """Station x-y (ns, 2) and the source on the fluid per unit span at the last output
    step, absolute pressure (ns, 2) [N/m], as in the old binTargets."""
    geo = json.load(open(os.path.join(runDir, 'geometry.json')))
    span = (1 + 2 * geo['zOverhang']) * geo['dz']
    X = readSurfaceNodes(os.path.join(runDir, 'embeddedSurface.top'))
    _, Fs = readSurfaceForce(os.path.join(runDir, 'postpro', 'EmbeddedSurfaceForce'))
    xy, station = np.unique(np.round(X[:, :2], 12), axis=0, return_inverse=True)
    Fst = np.zeros((len(xy), 2))
    np.add.at(Fst, station.ravel(), Fs[-1, :, :2])
    _, _, pInf = common.readInlet(os.path.join(runDir, 'FluidFile'))
    s = -(Fst - pInf * span * stationNormals(xy)) / span
    return xy, s, geo


def readSurfaceNodes(path):
    nodes = []
    with open(path) as f:
        f.readline()
        for line in f:
            s = line.split()
            if s[0] == 'Elements':
                break
            nodes.append([float(x) for x in s[1:4]])
    return np.array(nodes)


# ============= Main =================================================================
if __name__ == '__main__':
    old, _ = common.readTop(domainFile)
    xs, ys = np.unique(old[:, 0]), np.unique(old[:, 1])
    h = xs[1] - xs[0]
    hz = h
    nodes, tets = extrudedMesh(xs, ys, hz, nLayers)
    level = np.rint(nodes[:, 2] / hz).astype(int)
    thick = nLayers * hz
    print(f'coarse 3D mesh: {len(xs)} x {len(ys)} x {nLayers + 1} = {len(nodes)} nodes, '
          f'{len(tets)} tets, h = hz = {h:.4f}')

    xy, s, geo = stationSources(runDir)
    L = lengthsInCells(xy, nodes, tets, thick, nLayers * nzSample)
    F = L.T @ s                                                    # (nn, 2) force [N]
    print(f'\n1. sum_K F = {F.sum(axis=0)} N,  thickness x sum_s s = '
          f'{thick * s.sum(axis=0)} N')

    # 2. interior cells per unit span vs the old columns
    tg = np.load(os.path.join(runDir, 'targets.npz'))
    Sold, cols = tg['S'][-1], tg['columns']
    hit = np.flatnonzero(np.abs(F).sum(axis=1) > 0)
    print(f'\n2. cells hit: {len(hit)} on levels {sorted(set(level[hit]))}; '
          f'{len(cols)} old columns')
    print(f'   {"x, y":>16}  {"old Sx":>10} {"old Sy":>10} | per level k = 0..{nLayers}: '
          f'Sx = F_x / hz  [N/m]  (levels 0, {nLayers} are half cells)')
    worst = 0.0
    for c, (cx, cy) in enumerate(cols):
        at = np.flatnonzero(np.hypot(nodes[:, 0] - cx, nodes[:, 1] - cy) < 1e-6)
        at = at[np.argsort(level[at])]
        S3 = F[at] / hz
        interior = S3[1:-1]
        worst = max(worst, np.abs(interior - Sold[c]).max() / np.abs(Sold).max())
        print(f'   {cx:7.3f} {cy:7.3f}  {Sold[c, 0]:10.3e} {Sold[c, 1]:10.3e} | '
              + '  '.join(f'{v:10.3e}' for v in S3[:, 0]))
    print(f'   max |interior S - old S| / max |old S| = {worst:.1e}')

    # 3. volume fraction
    D = geo['D']
    u = (np.arange(nDisk) + 0.5) / nDisk - 0.5
    X, Y = np.meshgrid(geo['xc'] + D * u, geo['yc'] + D * u)
    disk = np.column_stack([X.ravel(), Y.ravel()])
    disk = disk[np.hypot(disk[:, 0] - geo['xc'], disk[:, 1] - geo['yc']) < D / 2]
    Vsolid = lengthsInCells(disk, nodes, tets, thick, nLayers * 16).sum(axis=0) * (D / nDisk) ** 2
    volFrac = Vsolid / dualVolumes(nodes, tets)
    inp = np.load(os.path.join(runDir, 'inputs.npz'))
    worstV = 0.0
    for c, (cx, cy) in enumerate(cols):
        at = np.flatnonzero(np.hypot(nodes[:, 0] - cx, nodes[:, 1] - cy) < 1e-6)
        at = at[(level[at] > 0) & (level[at] < nLayers)]
        oc = np.argmin(np.linalg.norm(inp['columns'] - [cx, cy], axis=1))
        worstV = max(worstV, np.abs(volFrac[at] - inp['volFrac'][oc]).max())
    print(f'\n3. interior volFrac vs old column volFrac: max abs difference {worstV:.1e}')
