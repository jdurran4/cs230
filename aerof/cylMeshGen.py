"""
Generate a quasi-2D embedded-cylinder problem for AERO-F.

Writes two files:
  domain.top          - coarse background fluid mesh (before any AMR). A structured
                        nx x ny grid in x-y, one cell thick in z, each hex split into
                        6 tets. Farfield on the x/y faces, symmetry on the z faces.
  embeddedSurface.top - the cylinder surface, a closed polygon extruded in z. It
                        overhangs both symmetry planes slightly so the surface does
                        not intersect the fluid mesh exactly on the boundary faces.

Formats follow the lab's parachute setup (JournalPaper/FSI_NAS):
  Nodes FluidNodes
  <nodeid> x y z
  Elements FluidMesh_0 using FluidNodes
  <elemid> 5 n1 n2 n3 n4                (type 5 = 4-node tet)
  Elements InletFixedSurface_3 using FluidNodes
  <elemid> 4 n1 n2 n3                   (type 4 = 3-node triangle)

The trailing _<id> on each surface group is the id referenced by SurfaceData[id]
(and by the AMR IntersectionCriterion FaceID) in the fluid input file.

Flow is in +x (Alpha = 0, Beta = 0 in the fluid file).

Usage: edit the parameters below, then run the script (e.g. play in VS Code).
"""
import math
import os

# ============= User-defined parameters ==============================================
plot_mesh = True    # plot the mesh and save mesh.png (requires matplotlib)
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sources')  # aerof/sources

D = 1.0             # cylinder diameter
xc, yc = -3.3, 0.1  # cylinder center
Lx, Ly = 20.0, 20.0 # domain size in x and y, centered on (x0, y0)
x0, y0 = 0.0, 0.0   # domain center
h = D / 1.2           # coarse mesh spacing in x and y
dz = 0.1            # domain thickness (one cell in z). Independent of h: the quasi-2D
                    # physics does not depend on it, but AMR refines across it too
                    # (Refine3D = 1 / Isotropic), so a thick layer multiplies the cost.
                    # ~ the coarsest AMR edge length (capsule MinimumEdgeLength = 0.1).
nTheta = 200        # number of points around the cylinder surface
zOverhang = 0.25    # embedded surface extends this fraction of dz past each z plane

# Surface ids (the _<id> suffix). Kept unique across fluid and embedded surfaces.
ID_ZMIN, ID_ZMAX = 1, 2            # symmetry planes
ID_XMIN, ID_XMAX = 3, 4            # farfield (x-min is the inflow side)
ID_YMIN, ID_YMAX = 5, 6            # farfield
ID_CYL = 7                         # embedded cylinder

FLUID_GROUPS = {  # boundary plane -> group name
    'zmin': f'SymmetrySurface_{ID_ZMIN}',
    'zmax': f'SymmetrySurface_{ID_ZMAX}',
    'xmin': f'InletFixedSurface_{ID_XMIN}',
    'xmax': f'InletFixedSurface_{ID_XMAX}',
    'ymin': f'InletFixedSurface_{ID_YMIN}',
    'ymax': f'InletFixedSurface_{ID_YMAX}',
}
CYL_GROUP = f'StickFixed_Cylinder_{ID_CYL}'

FMT = '%.16e'  # coordinate precision


# ============= Fluid domain =========================================================
def fluidNodes():
    """Structured grid nodes. x varies fastest, then y, then z (same as the FSI_NAS
    domain.top): id = 1 + i + nx*j + nx*ny*k."""
    nx = round(Lx / h) + 1
    ny = round(Ly / h) + 1
    if abs((nx - 1) * h - Lx) > 1e-12 * Lx or abs((ny - 1) * h - Ly) > 1e-12 * Ly:
        raise ValueError('Lx and Ly must be integer multiples of h')
    xs = [x0 - Lx / 2 + i * h for i in range(nx)]
    ys = [y0 - Ly / 2 + j * h for j in range(ny)]
    zs = [0.0, dz]
    nodes = [(x, y, z) for z in zs for y in ys for x in xs]
    return nx, ny, nodes


def signedVolume6(a, b, c, d):
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    w = [d[i] - a[i] for i in range(3)]
    return (u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))


def fluidTets(nx, ny, nodes):
    """Split each hex into 6 tets sharing the c000-c111 diagonal. Every hex uses the
    same split, so face diagonals conform between neighbors. Tets are ordered to have
    positive volume."""
    nid = lambda i, j, k: 1 + i + nx * j + nx * ny * k
    tets = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            c000, c100 = nid(i, j, 0), nid(i + 1, j, 0)
            c010, c110 = nid(i, j + 1, 0), nid(i + 1, j + 1, 0)
            c001, c101 = nid(i, j, 1), nid(i + 1, j, 1)
            c011, c111 = nid(i, j + 1, 1), nid(i + 1, j + 1, 1)
            for t in ((c000, c100, c110, c111), (c000, c100, c101, c111),
                      (c000, c010, c110, c111), (c000, c010, c011, c111),
                      (c000, c001, c011, c111), (c000, c001, c101, c111)):
                if signedVolume6(*(nodes[n - 1] for n in t)) < 0:
                    t = (t[0], t[2], t[1], t[3])
                tets.append(t)
    return tets


def boundaryFaces(nodes, tets):
    """Collect tet faces lying on each boundary plane, wound with outward normals.
    Taking them from the tets guarantees they match the volume mesh."""
    xmin, xmax = x0 - Lx / 2, x0 + Lx / 2
    ymin, ymax = y0 - Ly / 2, y0 + Ly / 2
    tol = 1e-9 * h
    planes = {  # name -> (coordinate index, value, outward sign)
        'zmin': (2, 0.0, -1), 'zmax': (2, dz, 1),
        'xmin': (0, xmin, -1), 'xmax': (0, xmax, 1),
        'ymin': (1, ymin, -1), 'ymax': (1, ymax, 1),
    }
    faces = {name: [] for name in planes}
    for t in tets:
        for f in ((t[0], t[1], t[2]), (t[0], t[1], t[3]),
                  (t[0], t[2], t[3]), (t[1], t[2], t[3])):
            p = [nodes[n - 1] for n in f]
            for name, (c, val, sgn) in planes.items():
                if all(abs(q[c] - val) < tol for q in p):
                    u = [p[1][i] - p[0][i] for i in range(3)]
                    v = [p[2][i] - p[0][i] for i in range(3)]
                    normal = [u[1] * v[2] - u[2] * v[1],
                              u[2] * v[0] - u[0] * v[2],
                              u[0] * v[1] - u[1] * v[0]]
                    if normal[c] * sgn < 0:
                        f = (f[0], f[2], f[1])
                    faces[name].append(f)
                    break
    return faces


def writeDomain(path, nodes, tets, faces):
    with open(path, 'w') as out:
        out.write('Nodes FluidNodes\n')
        for n, (x, y, z) in enumerate(nodes, start=1):
            out.write(f'{n}  {FMT % x}  {FMT % y}  {FMT % z}\n')
        elemId = 1
        out.write('Elements FluidMesh_0 using FluidNodes\n')
        for t in tets:
            out.write(f'{elemId}  5  {t[0]}  {t[1]}  {t[2]}  {t[3]}\n')
            elemId += 1
        for name in ('zmin', 'zmax', 'xmin', 'xmax', 'ymin', 'ymax'):
            out.write(f'Elements {FLUID_GROUPS[name]} using FluidNodes\n')
            for f in faces[name]:
                out.write(f'{elemId}  4  {f[0]}  {f[1]}  {f[2]}\n')
                elemId += 1


# ============= Embedded cylinder surface ============================================
def cylinderSurface():
    """Closed polygon of nTheta points, extruded from z = -zOverhang*dz to
    (1 + zOverhang)*dz. Points are offset by half a segment so no surface vertex sits
    on a fluid grid line of symmetry. Triangles are wound with outward normals.
    Node ids: bottom ring 1..nTheta, top ring nTheta+1..2*nTheta."""
    R = D / 2
    zb, zt = -zOverhang * dz, (1 + zOverhang) * dz
    ring = [(xc + R * math.cos(2 * math.pi * (k + 0.5) / nTheta),
             yc + R * math.sin(2 * math.pi * (k + 0.5) / nTheta)) for k in range(nTheta)]
    nodes = [(x, y, zb) for x, y in ring] + [(x, y, zt) for x, y in ring]
    tris = []
    for k in range(nTheta):
        b0, b1 = k + 1, (k + 1) % nTheta + 1
        t0, t1 = b0 + nTheta, b1 + nTheta
        tris.append((b0, b1, t1))
        tris.append((b0, t1, t0))
    return ring, nodes, tris


def writeEmbedded(path, nodes, tris):
    with open(path, 'w') as out:
        out.write('Nodes EmbeddedNodes\n')
        for n, (x, y, z) in enumerate(nodes, start=1):
            out.write(f'{n}  {FMT % x}  {FMT % y}  {FMT % z}\n')
        out.write(f'Elements {CYL_GROUP} using EmbeddedNodes\n')
        for e, t in enumerate(tris, start=1):
            out.write(f'{e}  4  {t[0]}  {t[1]}  {t[2]}\n')


def minNodeToSurfaceDistance(fluid, ring):
    """Smallest x-y distance from a fluid node to the cylinder polygon. A fluid node
    lying on (or extremely near) the embedded surface can upset the intersector."""
    best = float('inf')
    reach = D / 2 + 2 * h
    near = [(x, y) for x, y, _ in fluid if math.hypot(x - xc, y - yc) < reach]
    for k in range(len(ring)):
        (ax, ay), (bx, by) = ring[k], ring[(k + 1) % len(ring)]
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey
        for px, py in near:
            s = max(0.0, min(1.0, ((px - ax) * ex + (py - ay) * ey) / L2))
            best = min(best, math.hypot(px - ax - s * ex, py - ay - s * ey))
    return best


# ============= Plotting =============================================================
def plot(nodes, faces, ring, path):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    segs = [[nodes[a - 1][:2], nodes[b - 1][:2]]
            for f in faces['zmin'] for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, zoom in zip(axes, (None, 3 * D)):
        ax.add_collection(LineCollection(segs, colors='0.6', linewidths=0.5))
        ax.fill(*zip(*ring), color='tab:red', alpha=0.4, lw=0)
        ax.plot(*zip(*(ring + ring[:1])), color='tab:red', lw=1)
        ax.set_aspect('equal')
        if zoom is None:
            ax.set_xlim(x0 - Lx / 2, x0 + Lx / 2)
            ax.set_ylim(y0 - Ly / 2, y0 + Ly / 2)
            ax.set_title('Coarse mesh (z = 0 plane)')
            ax.annotate('', xy=(x0 - Lx / 2 + 0.15 * Lx, y0 + 0.4 * Ly),
                        xytext=(x0 - Lx / 2 + 0.03 * Lx, y0 + 0.4 * Ly),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
            ax.text(x0 - Lx / 2 + 0.03 * Lx, y0 + 0.42 * Ly, 'flow')
        else:
            ax.set_xlim(xc - zoom, xc + zoom)
            ax.set_ylim(yc - zoom, yc + zoom)
            ax.plot([p[0] for p in nodes], [p[1] for p in nodes], 'k.', ms=3)
            ax.set_title('Near cylinder')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'Wrote {path}')
    plt.show()


# ============= Main =================================================================
if __name__ == '__main__':
    os.makedirs(outdir, exist_ok=True)

    nx, ny, fluid = fluidNodes()
    tets = fluidTets(nx, ny, fluid)
    faces = boundaryFaces(fluid, tets)
    writeDomain(os.path.join(outdir, 'domain.top'), fluid, tets, faces)

    ring, cylNodes, cylTris = cylinderSurface()
    writeEmbedded(os.path.join(outdir, 'embeddedSurface.top'), cylNodes, cylTris)

    print(f'domain.top: {nx} x {ny} x 2 = {len(fluid)} nodes, {len(tets)} tets')
    for name in ('zmin', 'zmax', 'xmin', 'xmax', 'ymin', 'ymax'):
        print(f'  {FLUID_GROUPS[name]:22s} ({name}): {len(faces[name])} triangles')
    print(f'embeddedSurface.top: {CYL_GROUP}, {len(cylNodes)} nodes, {len(cylTris)} triangles,'
          f' edge length {math.pi * D / nTheta:.4g}')
    dmin = minNodeToSurfaceDistance(fluid, ring)
    print(f'min fluid node to cylinder distance: {dmin:.3e} ({dmin / h:.2e} h)')
    if dmin < 1e-3 * h:
        print('  WARNING: a fluid node nearly touches the embedded surface; '
              'consider offsetting the cylinder center slightly.')

    if plot_mesh:
        plot(fluid, faces, ring, os.path.join(outdir, 'mesh.png'))
