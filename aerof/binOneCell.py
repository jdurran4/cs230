"""
Phase 4 sanity check: bin the per-structure-node force Fs into coarse control volumes.

Coarse control volumes are the median-dual cells of the pre-AMR mesh (sources/domain.top).
The problem is quasi-2D (one tet layer in z), so the binning unit is a COLUMN: the dual
cells of grid point (i, j) on z = 0 and z = dz together. Binning by structure-node 3D
position would be meaningless here: both rings of structure nodes sit outside the fluid
slab (the tube overhangs it) and each surface triangle spans the full tube length.

Assignment of a structure node to columns: at the node's (x, y), sample z through the
slab, find the coarse tet containing each sample and take the vertex with the largest
barycentric coordinate (that is exactly the median-dual rule). The node's force is split
among columns by the fraction of samples landing in each. Because the hexes are split
into tets along a diagonal, the column footprint is not the h x h square around the grid
point; the square assignment is also reported for comparison.

Checks:
  1. sum over columns == sum over nodes (nothing lost)
  2. per-column table (time average over the second half of the run)
  3. for one column (default: largest |F|), compare against a modified-Newtonian
     estimate  dF = -q * Cp_max * max(0, -n.x)^2 * n * ds * span  (leeward Cp = 0)
  4. plot: coarse grid, cylinder nodes colored by column, column force arrows

Usage: set runName below, then run the script (e.g. play in VS Code).
"""
import os

import numpy as np

import checkSurfaceForce as chk
import cylMeshGen as mesh
import newRun

# ============= User-defined parameters ==============================================
runName = 'M1.5_Re100_x0.04_y0.12'
mach, Re = 1.5, 100        # must match the run (for the Newtonian estimate)
cell = None                # (x, y) of the column's grid point; None = largest |F|
nzSample = 64              # z samples per structure node for the dual-cell footprint
plot = True

aerofDir = os.path.dirname(os.path.abspath(__file__))
runDir = os.path.join(aerofDir, 'runs', runName)
domainFile = os.path.join(aerofDir, 'sources', 'domain.top')


# ============= Coarse mesh ==========================================================
def readDomain(path):
    """Returns nodes (nn, 3) and tets (ne, 4) as 0-based indices. Tets are the type-5
    elements (FluidMesh_0 in domain.top, Volume_0 in sower's OUTPUT.top.NNNN)."""
    nodes, tets, block = [], [], None
    with open(path) as f:
        for line in f:
            s = line.split()
            if s[0] in ('Nodes', 'Elements'):
                block = s[1]
                continue
            if block == 'FluidNodes':
                nodes.append([float(x) for x in s[1:4]])
            elif s[1] == '5':
                tets.append([int(n) - 1 for n in s[2:6]])
    return np.array(nodes), np.array(tets)


def owningNode(P, nodes, tets, chunk=4000):
    """Coarse node whose median-dual cell contains each point of P (np, 3): find the
    containing tet, then take the vertex with the largest barycentric coordinate."""
    Vall = nodes[tets]                                                      # (ne, 4, 3)
    Minv = np.linalg.inv((Vall[:, 1:] - Vall[:, :1]).transpose(0, 2, 1))    # (ne, 3, 3)
    tlo, thi = Vall[:, :, :2].min(axis=1), Vall[:, :, :2].max(axis=1)       # x-y bboxes

    order = np.lexsort((P[:, 1], P[:, 0]))      # spatially sorted chunks -> few tets each
    vtx = np.empty(len(P), dtype=int)
    for s in range(0, len(P), chunk):
        idx = order[s:s + chunk]
        p = P[idx]
        # Only tets whose x-y bounding box overlaps the chunk's can contain its points.
        near = np.all((thi >= p[:, :2].min(axis=0) - 1e-9)
                      & (tlo <= p[:, :2].max(axis=0) + 1e-9), axis=1)
        T, V, M = tets[near], Vall[near], Minv[near]
        lam123 = np.einsum('kij,pkj->pki', M, p[:, None, :] - V[None, :, 0])  # (p, k, 3)
        lam = np.concatenate([1 - lam123.sum(axis=2, keepdims=True), lam123], axis=2)
        inside = np.all(lam > -1e-12, axis=2)
        if not inside.any(axis=1).all():
            raise RuntimeError('some sample points are in no coarse tet')
        k = inside.argmax(axis=1)                                           # first hit
        vtx[idx] = T[k, lam[np.arange(len(p)), k].argmax(axis=1)]
    return vtx


def columnFractions(xy, nodes, tets, nz):
    """For each point in xy (np, 2), the fraction of the slab height belonging to each
    column. Returns (fractions (np, ncol), columns (ncol, 2) grid-point x-y)."""
    z = (np.arange(nz) + 0.5) / nz * mesh.dz
    P = np.column_stack([np.repeat(xy, nz, axis=0), np.tile(z, len(xy))])   # (np*nz, 3)
    vtx = owningNode(P, nodes, tets)

    colXY = np.round(nodes[vtx, :2], 9)
    columns, colId = np.unique(colXY, axis=0, return_inverse=True)
    frac = np.zeros((len(xy), len(columns)))
    np.add.at(frac, (np.repeat(np.arange(len(xy)), nz), colId.ravel()), 1.0 / nz)
    return frac, columns


def squareFractions(xy, columns):
    """Same, but with the h x h square around each grid point as the footprint."""
    frac = np.zeros((len(xy), len(columns)))
    for c, (cx, cy) in enumerate(columns):
        frac[:, c] = (np.abs(xy[:, 0] - cx) < mesh.h / 2) & (np.abs(xy[:, 1] - cy) < mesh.h / 2)
    return frac


# ============= Main =================================================================
if __name__ == '__main__':
    tF, Fs = chk.readSurfaceForce(os.path.join(runDir, 'postpro', 'EmbeddedSurfaceForce'))
    geo = mesh.loadGeometry(runDir)
    X = chk.readSurfaceNodes(geo['surfFile'])
    nodes, tets = readDomain(domainFile)

    # Quasi-2D: rings of nodes share x-y. Combine each (x, y) station's nodes.
    xy, station = np.unique(np.round(X[:, :2], 12), axis=0, return_inverse=True)
    station = station.ravel()
    Fst = np.zeros((len(tF), len(xy), 3))
    np.add.at(Fst, (slice(None), station), Fs)                              # (nt, ns, 3)
    tAvg = tF > 0.5 * tF[-1]
    FstAvg = Fst[tAvg].mean(axis=0)                                         # (ns, 3)

    frac, columns = columnFractions(xy, nodes, tets, nzSample)
    fracSq = squareFractions(xy, columns)
    Fcol = np.einsum('sc,ts->tc', frac, Fst[..., 0]), np.einsum('sc,ts->tc', frac, Fst[..., 1])
    Fcol = np.stack(Fcol, axis=2)                                           # (nt, ncol, 2)
    FcolAvg = Fcol[tAvg].mean(axis=0)
    FsqAvg = fracSq.T @ FstAvg[:, :2]

    # 1. conservation
    print(f'{len(tF)} output steps, {len(X)} structure nodes -> {len(xy)} x-y stations, '
          f'{len(columns)} columns hit')
    print(f'\n1. max |sum_columns F - sum_nodes F| over time = '
          f'{np.abs(Fcol.sum(axis=1) - Fs[:, :, :2].sum(axis=1)).max():.2e} N  '
          f'(square footprint covers {fracSq.sum(axis=1).min():.0f}-'
          f'{fracSq.sum(axis=1).max():.0f} per station)')

    # 2. table
    total = FstAvg[:, :2].sum(axis=0)
    print(f'\n2. column forces on the BODY, 2nd-half average (total = {total} N)')
    print(f'   {"grid point x, y":>18}  {"stations":>8}  {"Fx":>10}  {"Fy":>10}  '
          f'{"Fx/total":>8}  {"Fx (square)":>11}  {"Fy (square)":>11}')
    order = np.argsort(-np.abs(FcolAvg[:, 0]))
    for c in order:
        print(f'   {columns[c, 0]:8.3f} {columns[c, 1]:8.3f}  {frac[:, c].sum():8.1f}  '
              f'{FcolAvg[c, 0]:10.3e}  {FcolAvg[c, 1]:10.3e}  {FcolAvg[c, 0] / total[0]:8.3f}  '
              f'{FsqAvg[c, 0]:11.3e}  {FsqAvg[c, 1]:11.3e}')

    # 3. one column vs modified Newtonian
    c = order[0] if cell is None else int(np.argmin(np.linalg.norm(columns - cell, axis=1)))
    fs = newRun.freestream(mach, Re)
    q = 0.5 * fs['rho'] * fs['U'] ** 2
    g = newRun.gamma
    p02_p1 = (((g + 1) ** 2 * mach ** 2) / (4 * g * mach ** 2 - 2 * (g - 1))) ** (g / (g - 1)) \
        * (1 - g + 2 * g * mach ** 2) / (g + 1)
    cpMax = (p02_p1 - 1) / (0.5 * g * mach ** 2)
    xcyc = xy.mean(axis=0)
    n = (xy - xcyc) / np.linalg.norm(xy - xcyc, axis=1, keepdims=True)      # out of body
    ds = np.pi * mesh.D / len(xy)
    span = geo['span']
    dFnewt = -q * cpMax * np.maximum(0.0, -n[:, :1]) ** 2 * n * ds * span   # (ns, 2)
    Fnewt = frac[:, c] @ dFnewt
    print(f'\n3. column at ({columns[c, 0]:.3f}, {columns[c, 1]:.3f}): '
          f'{(frac[:, c] > 0).sum()} stations, arc angles '
          f'{np.degrees(np.arctan2(n[frac[:, c] > 0, 1], n[frac[:, c] > 0, 0])).min():.0f} to '
          f'{np.degrees(np.arctan2(n[frac[:, c] > 0, 1], n[frac[:, c] > 0, 0])).max():.0f} deg')
    print(f'   AERO-F (avg)       F = ({FcolAvg[c, 0]:.4e}, {FcolAvg[c, 1]:.4e}) N')
    print(f'   modified Newtonian F = ({Fnewt[0]:.4e}, {Fnewt[1]:.4e}) N  '
          f'(Cp_max = {cpMax:.3f}, span = tube length = {span:.3f} m, leeward Cp = 0)')
    print(f'   Newtonian whole body Fx = {dFnewt[:, 0].sum():.4e} N  vs AERO-F {total[0]:.4e} N')
    print(f'   source term on the FLUID in this column = -F = '
          f'({-FcolAvg[c, 0]:.4e}, {-FcolAvg[c, 1]:.4e}) N')

    # 4. plot
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        cmap = plt.get_cmap('tab10')
        fig, ax = plt.subplots(figsize=(8, 7))
        g0 = columns.min(axis=0) - mesh.h
        g1 = columns.max(axis=0) + mesh.h

        # Dual mesh: owner of a fine x-y grid of points, at several heights in the slab.
        npx = 240
        gx = np.linspace(g0[0], g1[0], npx)
        gy = np.linspace(g0[1], g1[1], npx)
        GX, GY = np.meshgrid(gx, gy)
        zAll = (np.arange(16) + 0.5) / 16 * mesh.dz
        P = np.column_stack([np.tile(np.column_stack([GX.ravel(), GY.ravel()]), (len(zAll), 1)),
                             np.repeat(zAll, GX.size)])
        vtx = owningNode(P, nodes, tets).reshape(len(zAll), npx, npx)
        ownerXY = np.round(nodes[vtx, :2], 9)
        # column index for hit columns, -1 for the others (unique id kept for boundaries)
        allCols, gid = np.unique(ownerXY.reshape(-1, 2), axis=0, return_inverse=True)
        gid = gid.reshape(len(zAll), npx, npx)
        hit = np.full(len(allCols), -1)
        for c, xyc in enumerate(columns):
            hit[np.argmin(np.linalg.norm(allCols - xyc, axis=1))] = c

        # shade by the column owning most of the slab height at each x-y
        maj = np.apply_along_axis(lambda v: np.bincount(v).argmax(), 0, gid)
        rgba = np.ones((npx, npx, 4))
        for g in np.unique(maj):
            if hit[g] >= 0:
                rgba[maj == g] = cmap(hit[g])
        rgba[..., 3] = 0.25
        ax.imshow(rgba, origin='lower', extent=(g0[0], g1[0], g0[1], g1[1]),
                  interpolation='nearest', zorder=0)

        # outline of the shaded (majority) cells of the hit columns
        for g in np.unique(maj):
            if hit[g] >= 0:
                ax.contour(GX, GY, (maj == g).astype(float), levels=[0.5], colors='k',
                           linewidths=1.0, zorder=2)

        # primal mesh: grid lines and the face diagonals of the tet split
        xg = np.arange(-mesh.Lx / 2 + mesh.x0, mesh.Lx / 2 + mesh.x0 + 1e-9, mesh.h)
        yg = np.arange(-mesh.Ly / 2 + mesh.y0, mesh.Ly / 2 + mesh.y0 + 1e-9, mesh.h)
        xg = xg[(xg > g0[0] - mesh.h) & (xg < g1[0] + mesh.h)]
        yg = yg[(yg > g0[1] - mesh.h) & (yg < g1[1] + mesh.h)]
        for x in xg:
            ax.axvline(x, color='0.6', lw=0.8, zorder=1)
        for y in yg:
            ax.axhline(y, color='0.6', lw=0.8, zorder=1)
        for x in xg[:-1]:                      # c000-c110 (z = 0) and c001-c111 (z = dz)
            for y in yg[:-1]:
                ax.plot([x, x + mesh.h], [y, y + mesh.h], color='0.6', lw=0.6, zorder=1)
                # exact 2D median dual of the face triangles (identical on z = 0 and z = dz):
                # segments from each triangle centroid to its edge midpoints
                for tri in (((x, y), (x + mesh.h, y), (x + mesh.h, y + mesh.h)),
                            ((x, y), (x, y + mesh.h), (x + mesh.h, y + mesh.h))):
                    tri = np.array(tri)
                    cen = tri.mean(axis=0)
                    for a in range(3):
                        mid = 0.5 * (tri[a] + tri[(a + 1) % 3])
                        ax.plot([cen[0], mid[0]], [cen[1], mid[1]], color='tab:blue', ls='--',
                                lw=0.8, zorder=2)

        # cylinder stations colored by owning column, grid points, forces at arc centroids
        owner = frac.argmax(axis=1)
        ax.scatter(xy[:, 0], xy[:, 1], c=owner, cmap=cmap, vmin=0, vmax=9, s=10, zorder=3)
        ax.scatter(columns[:, 0], columns[:, 1], c=np.arange(len(columns)), cmap=cmap,
                   vmin=0, vmax=9, marker='s', s=50, edgecolors='k', zorder=4)
        arcCentroid = (frac.T @ xy) / frac.sum(axis=0)[:, None]
        scale = 0.6 * mesh.h / np.abs(FcolAvg).max()
        ax.quiver(arcCentroid[:, 0], arcCentroid[:, 1], FcolAvg[:, 0] * scale,
                  FcolAvg[:, 1] * scale, color=[cmap(c) for c in range(len(columns))],
                  angles='xy', scale_units='xy', scale=1, width=0.006, edgecolor='k',
                  linewidth=0.5, zorder=5)

        ax.legend(handles=[
            Line2D([], [], color='0.6', lw=0.8, label='primal (coarse) mesh'),
            Line2D([], [], color='tab:blue', ls='--', lw=0.8,
                   label='dual cells on the z = 0 and z = dz faces'),
            Line2D([], [], color='k', lw=1.0,
                   label='dual cells, owner of most of the slab height (shaded)'),
        ], loc='upper left', fontsize=8, framealpha=0.9)
        ax.set_xlim(g0[0], g1[0])
        ax.set_ylim(g0[1], g1[1])
        ax.set_aspect('equal')
        ax.set_title(f'{runName}: shading = dual cell owning most of the slab at that x-y;\n'
                     f'arrows = time-avg force on the body piece in each cell, '
                     f'drawn from the piece centroid', fontsize=9)
        out = os.path.join(runDir, 'binOneCell.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'\n4. plot saved to {out}')
