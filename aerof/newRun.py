"""
Set up AERO-F run directories for the quasi-2D embedded cylinder.

Each case is one steady run in runs/<case>/: it pre-refines the coarse mesh (AMR
PreRefine), then solves steady from freestream with Hessian AMR. The converged steady
solution is the data sample (the cylinder stays steady at these Mach/Re, and the model
is quasi-steady); binTargets/binInputs read runs/<case>/postpro/.

Every case directory gets a FluidFile from FluidFile_template (with the substitutions in
substitutions()), the scripts in runTemplate/ (run.sh, clean.sh, postpro.sh, *.sbatch),
its own cylinder (embeddedSurface.top, geometry.json) and empty results/ references/
postpro/. Run one case with runs/<case>/run.sh NP, or all of them with ./runSweep.sh.

Everything else about the solver setup lives in FluidFile_template and is edited by
hand there.

The run is Dimensional (SI units). The freestream state is set from Mach, Re and a
fixed freestream temperature:
    a   = sqrt(gamma R T)         U = Mach a
    mu  = Sutherland(T)           rho = Re mu / (U D)        p = rho R T

D is imported from cylMeshGen.py so the input file always matches the generated mesh.

Cylinder position: every case shares the coarse fluid mesh (data/OUTPUT); only the
cylinder moves. A case's offset (ox, oy) places the center at baseNode + (ox, oy) h, so
the sweep samples where the body sits inside a coarse cell (the NN inputs offset/h and
volume fraction depend on it). Flow is along +x, so mirror symmetry in y means oy only
needs [0, 0.5] (mirror the samples in Python); ox needs [0, 1). Each case directory gets
its own embeddedSurface.top and geometry.json (cylMeshGen.writeCaseGeometry), and the
AMR capsules follow the cylinder.

Usage: edit the parameters below, then run the script (e.g. play in VS Code).
"""
import itertools
import math
import os
import re
import shutil
from collections import namedtuple

import cylMeshGen as mesh

# ============= User-defined parameters ==============================================
Case = namedtuple('Case', 'mach Re ox oy zOverhang maxElements',
                  defaults=(mesh.zOverhang, 1000000))  # AMR MaximumNumberOfElements

flows = [(m, Re) for Re in (20, 50, 100, 200)    # (Mach, Reynolds number based on D),
         for m in (1.2, 1.5, 2.0, 2.5)]        # Re ascending: Re 200 may not be steady
# 4 offsets per flow: spans ox and oy, two of each M1.5 Re100 Cd group (1.68 / 1.76,
# a checkerboard over the 3 x 3 grid below, cause unknown)
flowOffsets = [(0.14, 0.00), (0.50, 0.16), (0.86, 0.16), (0.50, 0.50)]
offsets = list(itertools.product(   # cylinder center offset (ox, oy) from baseNode, in h
    (0.14, 0.50, 0.86),             # ox in [0, 1)
    (0.00, 0.16, 0.50),             # oy in [0, 0.5] (mirror symmetry in y)
))                                  # every coarse node >= 0.075 h from the surface (the
                                    # ring, R = 0.6 h, passes near nodes for many offsets)
# Element-cap test (run first): does the M1.5 Re100 Cd split come from the truncated AMR?
# Repeats a 1.68 (x0.14_y0.00) and a 1.76 (x0.50_y0.00) case with twice the cap.
cases = [Case(1.5, 100, 0.14, 0.00, maxElements=2000000),
         Case(1.5, 100, 0.50, 0.00, maxElements=2000000)]
cases += [Case(m, Re, ox, oy) for m, Re in flows for ox, oy in flowOffsets]
# M1.5 Re100 on the full 3 x 3 offset grid (run 2026-10-02)
cases += [Case(1.5, 100, ox, oy) for ox, oy in offsets
          if (ox, oy) not in flowOffsets]
# The original position (0.04, 0.12), and the span test there: Fx should scale with the
# tube length (1 + 2 zOverhang) dz, i.e. by 1.2 / 1.5 from zOverhang 0.25 to 0.1.
cases += [Case(1.5, 100, 0.04, 0.12), Case(1.5, 100, 0.04, 0.12, zOverhang=0.1)]
overwrite = True  # True: rewrite FluidFile and scripts of existing run directories
                   # (outputs in results/ references/ postpro/ are left alone). An
                   # unchanged FluidFile is not rewritten, so DONE stays newer than it.

aerofDir = os.path.dirname(os.path.abspath(__file__))
runsDir = os.path.join(aerofDir, 'runs')                  # one subdirectory per case
templateFile = os.path.join(aerofDir, 'FluidFile_template')
templateDir = os.path.join(aerofDir, 'runTemplate')       # run scripts to copy
caseScripts = ('run.sh', 'clean.sh', 'postpro.sh', 'run.sbatch', 'postpro.sbatch')

# Coarse grid point the offsets are measured from: the one nearest the original center
baseNode = (mesh.x0 - mesh.Lx / 2 + round((mesh.xc - mesh.x0 + mesh.Lx / 2) / mesh.h) * mesh.h,
            mesh.y0 - mesh.Ly / 2 + round((mesh.yc - mesh.y0 + mesh.Ly / 2) / mesh.h) * mesh.h)
minNodeDistance = 0.02  # warn if a coarse node is closer than this (in h) to the surface

# Freestream gas
T_inf = 288.15          # freestream temperature [K]
gamma = 1.4             # ratio of specific heats (must match SpecificHeatRatio in template)
R_gas = 287.1           # ideal gas constant [J/(kg K)] (AERO-F default)
C_suth = 1.458e-6       # Sutherland constant [kg/(m s K^0.5)] (AERO-F default)
S_suth = 110.6          # Sutherland reference temperature [K] (AERO-F default, IoDataCore.C)

steadyMaxIts = 1000     # pseudo-time iterations (stops earlier at Time/Eps in template)


# ============= Freestream state =====================================================
def freestream(mach, Re):
    a = math.sqrt(gamma * R_gas * T_inf)
    U = mach * a
    mu = C_suth * T_inf ** 1.5 / (T_inf + S_suth)
    rho = Re * mu / (U * mesh.D)
    p = rho * R_gas * T_inf
    return dict(a=a, U=U, mu=mu, rho=rho, p=p)


def caseName(c):
    name = f'M{c.mach:g}_Re{c.Re:g}_x{c.ox:.2f}_y{c.oy:.2f}'
    name += f'_zo{c.zOverhang:g}' if c.zOverhang != mesh.zOverhang else ''
    return name + (f'_me{c.maxElements / 1e6:g}M' if c.maxElements != Case._field_defaults['maxElements']
                   else '')


def cylinderCenter(c):
    return baseNode[0] + c.ox * mesh.h, baseNode[1] + c.oy * mesh.h


# ============= Template substitution ================================================
def quoted(path, fromDir):
    return '"' + os.path.relpath(path, fromDir) + '"'


def substitutions(caseDir, c):
    """(block path, key) -> new value; paths relative to caseDir. Only direct children
    of the block are matched, so Time/MaxIts does not touch the Newton or linear solver
    MaxIts."""
    fs = freestream(c.mach, c.Re)
    xc, yc = cylinderCenter(c)
    inp = ('Input',)
    subs = {
        (inp, 'GeometryPrefix'): quoted(os.path.join(aerofDir, 'data', 'OUTPUT'), caseDir),
        (inp, 'EmbeddedSurface'): quoted(os.path.join(caseDir, 'embeddedSurface.top'), caseDir),
        (('BoundaryConditions', 'Inlet'), 'Mach'): f'{c.mach:.10g}',
        (('BoundaryConditions', 'Inlet'), 'Density'): f'{fs["rho"]:.10e}',
        (('BoundaryConditions', 'Inlet'), 'Pressure'): f'{fs["p"]:.10e}',
        (('Time',), 'MaxIts'): f'{steadyMaxIts}',
        (('AdaptiveMeshRefinement',), 'MaximumNumberOfElements'): f'{c.maxElements:d}',
    }
    # capsule axes start upstream of the cylinder center: [0] by 1.5 D, [2] and [3] by 3 D
    for crit, upstream in (('SelectionCriterion[0]', 1.5), ('SelectionCriterion[2]', 3.0),
                           ('SelectionCriterion[3]', 3.0)):
        path = ('AdaptiveMeshRefinement', crit, 'RefinementDomain', 'Capsule')
        subs[(path, 'Point_x')] = f'{xc - upstream * mesh.D:.10g}'
        subs[(path, 'Point_y')] = f'{yc:.10g}'
        subs[(path, 'Point_z')] = f'{mesh.dz / 2:.10g}'                # fluid mid-plane
    return subs


def caseInfo(c):
    fs = freestream(c.mach, c.Re)
    xc, yc = cylinderCenter(c)
    return f"""\
// Case {caseName(c)}: Mach = {c.mach:g}, Re_D = {c.Re:g}   (written by newRun.py)
//   cylinder center ({xc:.6g}, {yc:.6g}) = grid point ({baseNode[0]:.6g}, {baseNode[1]:.6g})
//   + ({c.ox:g}, {c.oy:g}) h;  zOverhang = {c.zOverhang:g} dz
//   D = {mesh.D:g} m, T = {T_inf:g} K, a = {fs['a']:.6g} m/s, U = {fs['U']:.6g} m/s
//   mu = {fs['mu']:.6g} Pa s, rho = {fs['rho']:.6g} kg/m^3, p = {fs['p']:.6g} Pa
//   D/U = {mesh.D / fs['U']:.6g} s, steady, MaxIts = {steadyMaxIts}"""


def fillTemplate(text, subs, info):
    """Replace `key = value;` lines at the given block paths. Every substitution must
    match exactly once, so a template edit can't silently break a sweep."""
    counts = {k: 0 for k in subs}
    stack = []
    out = []
    for line in text.splitlines():
        code = line.split('//', 1)[0]
        m = re.match(r'\s*under\s+(\w+)(\[\d+\])?\s*\{', code)
        if m:
            stack.append(m.group(1) + (m.group(2) or ''))
        else:
            m = re.match(r'(\s*)(\w+)(\s*=\s*)([^;]*);', code)
            if m and (tuple(stack), m.group(2)) in subs:
                k = (tuple(stack), m.group(2))
                counts[k] += 1
                line = f'{m.group(1)}{m.group(2)}{m.group(3)}{subs[k]};'
            stack = stack[:len(stack) - code.count('}')]
        out.append(line)
    bad = [f'{"/".join(p)}/{key} (found {n})' for (p, key), n in counts.items() if n != 1]
    if bad:
        raise ValueError('FluidFile_template: expected exactly one of ' + ', '.join(bad))
    text = '\n'.join(out) + '\n'
    if text.count('// CASE-INFO\n') != 1:
        raise ValueError('FluidFile_template: expected one "// CASE-INFO" header line')
    return text.replace('// CASE-INFO\n', info + '\n')


def setupRun(c):
    """Cylinder, FluidFile, run scripts and empty output directories for one case."""
    name = caseName(c)
    caseDir = os.path.join(runsDir, name)
    if os.path.exists(caseDir) and not overwrite:
        print(f'Skipped runs/{name}: already exists (set overwrite = True to rewrite)')
        return
    dmin = mesh.writeCaseGeometry(caseDir, *cylinderCenter(c), c.zOverhang)
    if dmin < minNodeDistance:
        print(f'WARNING runs/{name}: a coarse node is {dmin:.2e} h from the cylinder '
              f'surface; nudge the offset')
    for d in ('results', 'references', 'postpro'):
        os.makedirs(os.path.join(caseDir, d), exist_ok=True)
    for f in caseScripts:
        shutil.copy2(os.path.join(templateDir, f), caseDir)   # keeps the executable bit
    for sbatch in ('run.sbatch', 'postpro.sbatch'):
        sbatch = os.path.join(caseDir, sbatch)
        with open(sbatch) as fh:
            text = fh.read()
        with open(sbatch, 'w') as fh:
            fh.write(text.replace('--job-name=CASE', f'--job-name={name}'))

    with open(templateFile) as fh:
        text = fillTemplate(fh.read(), substitutions(caseDir, c), caseInfo(c))
    fluidFile = os.path.join(caseDir, 'FluidFile')
    old = open(fluidFile).read() if os.path.isfile(fluidFile) else None
    if text != old:
        with open(fluidFile, 'w') as fh:
            fh.write(text)
    status = ('new' if old is None else 'unchanged' if text == old
              else 'CHANGED' + (' (DONE is now stale: rerun)' if os.path.exists(
                  os.path.join(caseDir, 'DONE')) else ''))
    fs = freestream(c.mach, c.Re)
    print(f'runs/{name}: FluidFile {status}; min node-surface distance {dmin:.3f} h, '
          f'U = {fs["U"]:.4g} m/s, rho = {fs["rho"]:.4g} kg/m^3, p = {fs["p"]:.4g} Pa')


# ============= Main =================================================================
if __name__ == '__main__':
    for c in cases:
        setupRun(c)
