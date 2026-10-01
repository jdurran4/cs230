"""
Set up AERO-F run directories for the quasi-2D embedded cylinder.

Each case runs as two stages linked by a restart (the lab's FSI_NAS pattern):

  runs/M<Mach>_Re<Re>/steady/  pre-refines the coarse mesh (AMR PreRefine), then solves
                               steady from freestream
  runs/M<Mach>_Re<Re>/         production (unsteady) run restarted from the steady
                               solution on its refined mesh; its postpro/ is what
                               binTargets/binInputs read

Every stage directory gets a FluidFile from FluidFile_template (stage blocks selected by
"// #if <stage>" lines, plus the substitutions in substitutions()), the scripts in
runTemplate/ (run.sh, clean.sh, postpro.sh, *.sbatch) and empty results/ references/
postpro/. The case directory also gets runChain.sh, which runs steady/ then the
production run with the same NP (the production stage reads the refined mesh, which
has a CPU map only for the NP the steady stage ran with):
    runs/<case>/runChain.sh NP

Everything else about the solver setup lives in FluidFile_template and is edited by
hand there.

The run is Dimensional (SI units). The freestream state is set from Mach, Re and a
fixed freestream temperature:
    a   = sqrt(gamma R T)         U = Mach a
    mu  = Sutherland(T)           rho = Re mu / (U D)        p = rho R T
Time is specified in convective units D/U so a sweep keeps the same resolution of
the flow physics in every case (so output Frequency, in steps, can stay fixed in the
template).

D is imported from cylMeshGen.py so the input file always matches the generated mesh.

Usage: edit the parameters below, then run the script (e.g. play in VS Code).
"""
import math
import os
import re
import shutil

import cylMeshGen as mesh

# ============= User-defined parameters ==============================================
cases = [          # (Mach, Reynolds number based on D); add entries for a sweep
    (1.5, 100),
]
overwrite = True  # True: rewrite FluidFile and scripts of existing run directories
                   # (outputs in results/ references/ postpro/ are left alone)

aerofDir = os.path.dirname(os.path.abspath(__file__))
runsDir = os.path.join(aerofDir, 'runs')                  # one subdirectory per case
templateFile = os.path.join(aerofDir, 'FluidFile_template')
templateDir = os.path.join(aerofDir, 'runTemplate')       # run scripts to copy
stageScripts = ('run.sh', 'clean.sh', 'postpro.sh', 'run.sbatch', 'postpro.sbatch')

# Freestream gas
T_inf = 288.15          # freestream temperature [K]
gamma = 1.4             # ratio of specific heats (must match SpecificHeatRatio in template)
R_gas = 287.1           # ideal gas constant [J/(kg K)] (AERO-F default)
C_suth = 1.458e-6       # Sutherland constant [kg/(m s K^0.5)] (AERO-F default)
S_suth = 110.6          # Sutherland reference temperature [K] (AERO-F default, IoDataCore.C)

# Steady stage
steadyMaxIts = 1000     # pseudo-time iterations (stops earlier at Time/Eps in template)

# Production (unsteady) stage, in convective units D/U. It starts from the steady
# solution: a steady case only needs to show nothing grows; a shedding case needs the
# instability to grow plus several shedding periods (St ~ 0.2 -> period ~ 5 D/U).
dt_conv = 0.0255        # time step (~5e-5 s at M1.5, Re100)
tEnd_conv = 100.0      # final time (transient + collection window)


# ============= Freestream state =====================================================
def freestream(mach, Re):
    a = math.sqrt(gamma * R_gas * T_inf)
    U = mach * a
    mu = C_suth * T_inf ** 1.5 / (T_inf + S_suth)
    rho = Re * mu / (U * mesh.D)
    p = rho * R_gas * T_inf
    return dict(a=a, U=U, mu=mu, rho=rho, p=p)


def caseName(mach, Re):
    return f'M{mach:g}_Re{Re:g}'


# ============= Template substitution ================================================
def quoted(path, fromDir):
    return '"' + os.path.relpath(path, fromDir) + '"'


def substitutions(stage, stageDir, caseDir, mach, Re):
    """(block path, key) -> new value for one stage; paths relative to stageDir. Only
    direct children of the block are matched, so Time/MaxIts does not touch the Newton
    or linear solver MaxIts."""
    steadyRefs = os.path.join(caseDir, 'steady', 'references')
    fs = freestream(mach, Re)
    inp = ('Input',)
    subs = {
        (('Problem',), 'Type'): 'Unsteady' if stage == 'unsteady' else 'Steady',
        (inp, 'GeometryPrefix'): quoted(os.path.join(steadyRefs, 'OUTPUT') if stage == 'unsteady'
                                        else os.path.join(aerofDir, 'data', 'OUTPUT'), stageDir),
        (inp, 'EmbeddedSurface'): quoted(os.path.join(aerofDir, 'sources', 'embeddedSurface.top'),
                                         stageDir),
        (('BoundaryConditions', 'Inlet'), 'Mach'): f'{mach:.10g}',
        (('BoundaryConditions', 'Inlet'), 'Density'): f'{fs["rho"]:.10e}',
        (('BoundaryConditions', 'Inlet'), 'Pressure'): f'{fs["p"]:.10e}',
    }
    if stage == 'steady':
        subs[(('Time',), 'MaxIts')] = f'{steadyMaxIts}'
    else:
        tRef = mesh.D / fs['U']
        subs[(inp, 'MeshData')] = quoted(os.path.join(steadyRefs, 'MeshData'), stageDir)
        subs[(inp, 'Solution')] = quoted(os.path.join(steadyRefs, 'Solution'), stageDir)
        subs[(('Time',), 'MaxIts')] = f'{math.ceil(tEnd_conv / dt_conv) + 10}'
        subs[(('Time',), 'TimeStep')] = f'{dt_conv * tRef:.10e}'
        subs[(('Time',), 'MaxTime')] = f'{tEnd_conv * tRef:.10e}'
    return subs


def caseInfo(stage, mach, Re):
    fs = freestream(mach, Re)
    tRef = mesh.D / fs['U']
    time = (f'steady, MaxIts = {steadyMaxIts}' if stage == 'steady'
            else f'dt = {dt_conv:g} D/U, tEnd = {tEnd_conv:g} D/U')
    return f"""\
// Case {caseName(mach, Re)}, stage {stage}: Mach = {mach:g}, Re_D = {Re:g}   (written by newRun.py)
//   D = {mesh.D:g} m, T = {T_inf:g} K, a = {fs['a']:.6g} m/s, U = {fs['U']:.6g} m/s
//   mu = {fs['mu']:.6g} Pa s, rho = {fs['rho']:.6g} kg/m^3, p = {fs['p']:.6g} Pa
//   D/U = {tRef:.6g} s, {time}"""


def selectStage(text, stage):
    """Keep the lines of `// #if <stages>` ... `// #else` ... `// #endif` blocks that
    apply to this stage (blocks may nest); drop the directive lines."""
    keep = [True]          # stack: is the current block kept
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        d = line.strip()
        if d.startswith('// #if '):
            keep.append(keep[-1] and stage in d[len('// #if '):].split())
        elif d == '// #else':
            if len(keep) == 1:
                raise ValueError(f'FluidFile_template line {n}: #else without #if')
            keep[-1] = keep[-2] and not keep[-1]
        elif d == '// #endif':
            if len(keep) == 1:
                raise ValueError(f'FluidFile_template line {n}: #endif without #if')
            keep.pop()
        elif keep[-1]:
            out.append(line)
    if len(keep) != 1:
        raise ValueError('FluidFile_template: unclosed #if')
    return '\n'.join(out) + '\n'


def fillTemplate(text, stage, subs, info):
    """Select the stage's lines, then replace `key = value;` lines at the given block
    paths. Every substitution must match exactly once, so a template edit can't
    silently break a sweep."""
    text = selectStage(text, stage)
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
        raise ValueError(f'FluidFile_template ({stage} stage): expected exactly one of '
                         + ', '.join(bad))
    text = '\n'.join(out) + '\n'
    if text.count('// CASE-INFO\n') != 1:
        raise ValueError('FluidFile_template: expected one "// CASE-INFO" header line')
    return text.replace('// CASE-INFO\n', info + '\n')


def setupStage(stageDir, stage, jobName, caseDir, mach, Re):
    """FluidFile, run scripts and empty output directories for one stage."""
    os.makedirs(stageDir, exist_ok=True)
    for d in ('results', 'references', 'postpro'):
        os.makedirs(os.path.join(stageDir, d), exist_ok=True)
    for f in stageScripts:
        shutil.copy2(os.path.join(templateDir, f), stageDir)  # keeps the executable bit
    for sbatch in ('run.sbatch', 'postpro.sbatch'):
        sbatch = os.path.join(stageDir, sbatch)
        with open(sbatch) as fh:
            text = fh.read()
        with open(sbatch, 'w') as fh:
            fh.write(text.replace('--job-name=CASE', f'--job-name={jobName}'))

    with open(templateFile) as fh:
        text = fh.read()
    subs = substitutions(stage, stageDir, caseDir, mach, Re)
    with open(os.path.join(stageDir, 'FluidFile'), 'w') as fh:
        fh.write(fillTemplate(text, stage, subs, caseInfo(stage, mach, Re)))


def setupRun(mach, Re):
    name = caseName(mach, Re)
    runDir = os.path.join(runsDir, name)
    if os.path.exists(runDir) and not overwrite:
        print(f'Skipped runs/{name}: already exists (set overwrite = True to rewrite)')
        return
    setupStage(os.path.join(runDir, 'steady'), 'steady', f'{name}_steady', runDir, mach, Re)
    setupStage(runDir, 'unsteady', name, runDir, mach, Re)

    shutil.copy2(os.path.join(templateDir, 'runChain.sh'), runDir)

    fs = freestream(mach, Re)
    print(f'Wrote runs/{name} (+ steady/): U = {fs["U"]:.4g} m/s, rho = {fs["rho"]:.4g} kg/m^3, '
          f'p = {fs["p"]:.4g} Pa, dt = {dt_conv * mesh.D / fs["U"]:.4g} s')


# ============= Main =================================================================
if __name__ == '__main__':
    for mach, Re in cases:
        setupRun(mach, Re)
