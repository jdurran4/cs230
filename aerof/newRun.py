"""
Set up AERO-F run directories for the quasi-2D embedded cylinder.

For each (Mach, Re) case this creates aerof/runs/M<Mach>_Re<Re>/ containing
  FluidFile      copy of FluidFile_template with only the freestream and time lines
                 replaced (see substitutions())
  run.sh, run.sbatch, clean.sh, postpro.sh, postpro.sbatch   copied from runTemplate/
  results/ references/ postpro/              empty output directories

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

# Freestream gas
T_inf = 288.15          # freestream temperature [K]
gamma = 1.4             # ratio of specific heats (must match SpecificHeatRatio in template)
R_gas = 287.1           # ideal gas constant [J/(kg K)] (AERO-F default)
C_suth = 1.458e-6       # Sutherland constant [kg/(m s K^0.5)] (AERO-F default)
S_suth = 110.6          # Sutherland reference temperature [K] (AERO-F default, IoDataCore.C)

# Time, in convective units D/U
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
def substitutions(mach, Re):
    """(block path, key) -> new value. Only direct children of the block are matched,
    so Time/MaxIts does not touch the Newton or linear solver MaxIts."""
    fs = freestream(mach, Re)
    tRef = mesh.D / fs['U']
    return {
        (('BoundaryConditions', 'Inlet'), 'Mach'): f'{mach:.10g}',
        (('BoundaryConditions', 'Inlet'), 'Density'): f'{fs["rho"]:.10e}',
        (('BoundaryConditions', 'Inlet'), 'Pressure'): f'{fs["p"]:.10e}',
        (('Time',), 'MaxIts'): f'{math.ceil(tEnd_conv / dt_conv) + 10}',
        (('Time',), 'TimeStep'): f'{dt_conv * tRef:.10e}',
        (('Time',), 'MaxTime'): f'{tEnd_conv * tRef:.10e}',
    }


def caseInfo(mach, Re):
    fs = freestream(mach, Re)
    tRef = mesh.D / fs['U']
    return f"""\
// Case {caseName(mach, Re)}: Mach = {mach:g}, Re_D = {Re:g}   (written by newRun.py)
//   D = {mesh.D:g} m, T = {T_inf:g} K, a = {fs['a']:.6g} m/s, U = {fs['U']:.6g} m/s
//   mu = {fs['mu']:.6g} Pa s, rho = {fs['rho']:.6g} kg/m^3, p = {fs['p']:.6g} Pa
//   D/U = {tRef:.6g} s, dt = {dt_conv:g} D/U, tEnd = {tEnd_conv:g} D/U"""


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


def setupRun(mach, Re):
    name = caseName(mach, Re)
    runDir = os.path.join(runsDir, name)
    if os.path.exists(runDir) and not overwrite:
        print(f'Skipped runs/{name}: already exists (set overwrite = True to rewrite)')
        return
    os.makedirs(runDir, exist_ok=True)
    for d in ('results', 'references', 'postpro'):
        os.makedirs(os.path.join(runDir, d), exist_ok=True)

    for f in os.listdir(templateDir):
        shutil.copy2(os.path.join(templateDir, f), runDir)  # keeps the executable bit
    for sbatch in ('run.sbatch', 'postpro.sbatch'):
        sbatch = os.path.join(runDir, sbatch)
        with open(sbatch) as fh:
            text = fh.read()
        with open(sbatch, 'w') as fh:
            fh.write(text.replace('--job-name=CASE', f'--job-name={name}'))

    with open(templateFile) as fh:
        text = fh.read()
    with open(os.path.join(runDir, 'FluidFile'), 'w') as fh:
        fh.write(fillTemplate(text, substitutions(mach, Re), caseInfo(mach, Re)))

    fs = freestream(mach, Re)
    print(f'Wrote runs/{name}: U = {fs["U"]:.4g} m/s, rho = {fs["rho"]:.4g} kg/m^3, '
          f'p = {fs["p"]:.4g} Pa, dt = {dt_conv * mesh.D / fs["U"]:.4g} s')


# ============= Main =================================================================
if __name__ == '__main__':
    for mach, Re in cases:
        setupRun(mach, Re)
