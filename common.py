"""Shared paths, gas constants, and AERO-F file readers.

Data lives outside the repo (see README): DATA is the data root, STEADY the steady-v0
pass (resolved runs, coarse oracle runs, meshes) in its original layout, UNSTEADY the
unsteady sweep (mesh/, data/, runs/; set up by sweep/).
"""
import os
import re

import numpy as np

DATA = os.path.expanduser(os.environ.get('CS230_DATA', '~/Research/cs230-data'))
STEADY = os.path.join(DATA, 'steady-v0')
UNSTEADY = os.path.join(DATA, 'unsteady')

# Gas model of every run (AERO-F defaults; must match the FluidFiles)
gamma = 1.4          # ratio of specific heats
R_gas = 287.1        # J/(kg K)
C_suth = 1.458e-6    # Sutherland constant, kg/(m s K^0.5)
S_suth = 110.6       # Sutherland reference temperature, K


def viscosity(T):
    return C_suth * T ** 1.5 / (T + S_suth)


def readTop(path):
    """Nodes (nn, 3) and tets (ne, 4, 0-based) of an AERO-F .top mesh. Tets are the
    type-5 elements (FluidMesh_0 in domain.top, Volume_0 in sower's OUTPUT.top.NNNN)."""
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


def readXpost(path):
    """Times (nt,) and nodal values (nt, nn) or (nt, nn, 3) of an .xpost file."""
    with open(path) as f:
        kind = f.readline().split()[0]                    # Scalar or Vector
        nn = int(f.readline())
        data = np.array(f.read().split(), dtype=float)
    nc = 3 if kind == 'Vector' else 1
    per = 1 + nn * nc
    nt = len(data) // per                                 # drops a partial last step
    data = data[:nt * per].reshape(nt, per)
    vals = data[:, 1:].reshape(nt, nn, nc)
    return data[:, 0], vals[..., 0] if nc == 1 else vals


def readInlet(fluidFile):
    """Mach, Density, Pressure from the BoundaryConditions/Inlet block of a FluidFile."""
    text = open(fluidFile).read()
    block = re.search(r'under\s+Inlet\s*\{(.*?)\}', text, re.S).group(1)
    get = lambda key: float(re.search(rf'\b{key}\s*=\s*([^;]+);', block).group(1))
    return get('Mach'), get('Density'), get('Pressure')


def residualHistory(logFile):
    """Residuals of every nonlinear iteration in an AERO-F log, and whether it aborted."""
    log = open(logFile).read()
    res = [float(r) for r in re.findall(r'^It\s+\d+: Res = (\S+),', log, re.M)]
    return np.array(res), 'Aborting' in log
