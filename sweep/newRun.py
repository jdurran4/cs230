"""
Set up AERO-F run directories for the unsteady quasi-2D embedded cylinder sweep.

Each case is one unsteady run in unsteady/runs/<case>/: an impulsive start from
freestream, time-accurate for tEnd D/U, with snapshots every outputEvery D/U (see
FluidFile_template). The startup transient is the data.

Every case directory gets a FluidFile from FluidFile_template (with the substitutions in
substitutions()), the scripts in runTemplate/ (run.sh, clean.sh, postpro.sh, *.sbatch),
its own cylinder (embeddedSurface.top, geometry.json) and empty results/ references/
postpro/. Run one case with runs/<case>/run.sh NP, or all of them with sweep/runSweep.sh
(in the order of unsteady/runs.txt, which this script writes).

Everything else about the solver setup lives in FluidFile_template and is edited by
hand there.

The run is Dimensional (SI units). The freestream state is set from Mach, Re and a
fixed freestream temperature:
    a   = sqrt(gamma R T)         U = Mach a
    mu  = Sutherland(T)           rho = Re mu / (U D)        p = rho R T
Times are set in units of D/U, so every case takes the same number of steps.

Cylinder position: every case shares the coarse fluid mesh (unsteady/data/OUTPUT); only
the cylinder moves. A case's offset (ox, oy) places the center at baseNode + (ox, oy) h,
so the sweep samples where the body sits inside a coarse cell. Flow is along +x, so
mirror symmetry in y means oy only needs [0, 0.5]; ox needs [0, 1). The AMR capsules
follow the cylinder.

Usage (from the repo root):  python -m sweep.newRun
"""
import math
import os
import re
import shutil
from collections import namedtuple

import numpy as np

import common
from sweep import meshGen as mesh

# ============= User-defined parameters ==============================================
Case = namedtuple('Case', 'mach Re ox oy')

# Reference case first: the steady-v0 oracle/online test case (M1.5 Re100 at (0.04, 0.12))
cases = [Case(1.5, 100, 0.04, 0.12)]
# Latin hypercube in (Mach, log Re, ox, oy), nSweep cases
nSweep = 15
sweepSeed = 0
sweepMach = (1.2, 2.5)
sweepRe = (20, 200)          # sampled uniformly in log Re
sweepMinNodeDistance = 0.05  # redraw (ox, oy) if a coarse node is closer (in h)

# Time integration, in units of D/U (U = freestream speed of the case)
stepsPerDU = 20         # TimeStep = (D/U) / stepsPerDU
tEnd = 50.0             # MaxTime: the startup wave and wake reach the outlet (~47 D)
outputEvery = 0.25      # field + EmbeddedSurfaceForce snapshots (only these exist later)
amrEvery = 1.0          # Hessian AMR adaptation
restartEvery = 10.0     # restart files

overwrite = True  # True: rewrite FluidFile and scripts of existing run directories
                   # (outputs in results/ references/ postpro/ are left alone). An
                   # unchanged FluidFile is not rewritten, so DONE stays newer than it.

sweepDir = os.path.dirname(os.path.abspath(__file__))
runsDir = os.path.join(common.UNSTEADY, 'runs')           # one subdirectory per case
templateFile = os.path.join(sweepDir, 'FluidFile_template')
templateDir = os.path.join(sweepDir, 'runTemplate')       # run scripts to copy
caseScripts = ('run.sh', 'clean.sh', 'postpro.sh', 'run.sbatch', 'postpro.sbatch')

# Coarse grid point the offsets are measured from: the one nearest the nominal center
baseNode = (mesh.x0 - mesh.Lx / 2 + round((mesh.xc - mesh.x0 + mesh.Lx / 2) / mesh.h) * mesh.h,
            mesh.y0 - mesh.Ly / 2 + round((mesh.yc - mesh.y0 + mesh.Ly / 2) / mesh.h) * mesh.h)
minNodeDistance = 0.02  # warn if a coarse node is closer than this (in h) to the surface

T_inf = 288.15          # freestream temperature [K]


# ============= Continuous sweep =====================================================
def sweepCases():
    """nSweep Cases from a Latin hypercube: each of the 4 coordinates is split into
    nSweep equal strata, and each stratum is used exactly once (random pairing).
    Rounded so the case name reproduces the values exactly."""
    rng = np.random.default_rng(sweepSeed)
    u = (np.array([rng.permutation(nSweep) for _ in range(4)]).T
         + rng.random((nSweep, 4))) / nSweep                     # (nSweep, 4) in [0, 1)
    _, _, fluid = mesh.fluidNodes()
    out = []
    for uM, uRe, ux, uy in u:
        mach = round(sweepMach[0] + uM * (sweepMach[1] - sweepMach[0]), 3)
        Re = round(float(np.exp(np.log(sweepRe[0]) + uRe * np.log(sweepRe[1] / sweepRe[0]))), 1)
        ox, oy = round(ux, 2) % 1.0, round(0.5 * uy, 2)
        while True:   # keep coarse nodes off the surface (the intersector dislikes it)
            c = Case(mach, Re, ox, oy)
            ring = mesh.cylinderSurface(*cylinderCenter(c))[0]
            if mesh.minNodeToSurfaceDistance(fluid, ring, *cylinderCenter(c)) / mesh.h \
                    >= sweepMinNodeDistance:
                break
            ox, oy = round(rng.random(), 2) % 1.0, round(0.5 * rng.random(), 2)
        out.append(c)
    return out


# ============= Freestream state =====================================================
def freestream(mach, Re):
    a = math.sqrt(common.gamma * common.R_gas * T_inf)
    U = mach * a
    mu = common.viscosity(T_inf)
    rho = Re * mu / (U * mesh.D)
    p = rho * common.R_gas * T_inf
    return dict(a=a, U=U, mu=mu, rho=rho, p=p, tRef=mesh.D / U)


def caseName(c):
    return f'M{c.mach:g}_Re{c.Re:g}_x{c.ox:.2f}_y{c.oy:.2f}'


def cylinderCenter(c):
    return baseNode[0] + c.ox * mesh.h, baseNode[1] + c.oy * mesh.h


def steps(tDU):
    """Number of time steps in tDU units of D/U."""
    return round(tDU * stepsPerDU)


# ============= Template substitution ================================================
def quoted(path, fromDir):
    return '"' + os.path.relpath(path, fromDir) + '"'


def substitutions(caseDir, c):
    """(block path, key) -> new value; paths relative to caseDir. Only direct children
    of the block are matched, so Time/MaxIts does not touch the Newton or linear solver
    MaxIts."""
    fs = freestream(c.mach, c.Re)
    xc, yc = cylinderCenter(c)
    inp, amr, inlet = ('Input',), ('AdaptiveMeshRefinement',), ('BoundaryConditions', 'Inlet')
    subs = {
        (inp, 'GeometryPrefix'): quoted(os.path.join(common.UNSTEADY, 'data', 'OUTPUT'), caseDir),
        (inp, 'EmbeddedSurface'): quoted(os.path.join(caseDir, 'embeddedSurface.top'), caseDir),
        (inlet, 'Mach'): f'{c.mach:.10g}',
        (inlet, 'Density'): f'{fs["rho"]:.10e}',
        (inlet, 'Pressure'): f'{fs["p"]:.10e}',
        (('Time',), 'TimeStep'): f'{fs["tRef"] / stepsPerDU:.10e}',
        (('Time',), 'MaxTime'): f'{fs["tRef"] * tEnd:.10e}',
        (('Time',), 'MaxIts'): f'{steps(tEnd) + 100}',   # margin: the run stops at MaxTime
        (amr, 'Frequency'): f'{steps(amrEvery)}',
        (amr, 'BalancingFrequency'): f'{steps(amrEvery)}',
        (('Output', 'Postpro'), 'Frequency'): f'{steps(outputEvery)}',
        (('Output', 'Restart'), 'Frequency'): f'{steps(restartEvery)}',
    }
    # capsule axes start upstream of the cylinder center: [0] by 1.5 D, [2] and [3] by 3 D
    for crit, upstream in (('SelectionCriterion[0]', 1.5), ('SelectionCriterion[2]', 3.0),
                           ('SelectionCriterion[3]', 3.0)):
        path = amr + (crit, 'RefinementDomain', 'Capsule')
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
//   + ({c.ox:g}, {c.oy:g}) h
//   D = {mesh.D:g} m, T = {T_inf:g} K, a = {fs['a']:.6g} m/s, U = {fs['U']:.6g} m/s
//   mu = {fs['mu']:.6g} Pa s, rho = {fs['rho']:.6g} kg/m^3, p = {fs['p']:.6g} Pa
//   D/U = {fs['tRef']:.6g} s; dt = D/U / {stepsPerDU}, t_end = {tEnd:g} D/U ({steps(tEnd)} steps),
//   snapshots every {outputEvery:g} D/U, AMR every {amrEvery:g} D/U"""


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
                comment = line[len(code):]
                line = f'{m.group(1)}{m.group(2)}{m.group(3)}{subs[k]};' + \
                    (f'   {comment.strip()}' if comment.strip() else '')
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
    dmin = mesh.writeCaseGeometry(caseDir, *cylinderCenter(c), mesh.zOverhang)
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
          f'U = {fs["U"]:.4g} m/s, dt = {fs["tRef"] / stepsPerDU:.4g} s')


def runOrder(allCases):
    """Run order for runSweep.sh: the listed cases first, then greedily the sweep case
    farthest (in Mach and log Re, each scaled to [0, 1]) from those already chosen, so
    any prefix of the sweep spans the flow range."""
    def point(c):
        return np.array([(c.mach - sweepMach[0]) / (sweepMach[1] - sweepMach[0]),
                         np.log(c.Re / sweepRe[0]) / np.log(sweepRe[1] / sweepRe[0])])
    order, rest = list(cases), [c for c in allCases if c not in cases]
    while rest:
        far = max(rest, key=lambda c: min(np.linalg.norm(point(c) - point(o)) for o in order))
        order.append(far)
        rest.remove(far)
    return order


# ============= Main =================================================================
if __name__ == '__main__':
    os.makedirs(runsDir, exist_ok=True)
    # run scripts find tools.sh by walking up from the case directory
    shutil.copy2(os.path.join(sweepDir, 'tools.sh'), common.UNSTEADY)
    allCases = cases + sweepCases()
    for c in allCases:
        setupRun(c)
    with open(os.path.join(common.UNSTEADY, 'runs.txt'), 'w') as fh:
        fh.write(''.join(caseName(c) + '\n' for c in runOrder(allCases)))
    print(f'Wrote runs.txt (run order for sweep/runSweep.sh)')
