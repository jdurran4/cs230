"""
Unsteady online runs: the coarse oracle setup (sweep/oracle.py: extruded coarse mesh, no
cylinder, impulsive start, same time stepping) with the neural source in place of the
oracle file. The network predicts the source per unit span S of each column the body
touches; AERO-F applies S / h^2 per unit volume at every level of the column, as the
oracle does with the true source.

Writes unsteady/online/<case>_<tag>/:
  FluidFile       OracleFluidFile_template, OracleSourceFile replaced by
                  NeuralSourceWeightsFile (DATA/nn/<net>/nnWeights.txt) and
                  NeuralSourceGeometryFile
  nnGeometry.txt  gas constants, then per column: x y h D volFrac Nx Ny rx ry (the
                  interior-cell geometry, learn/dataset.columnGeometry). Sliver columns
                  with no solid volume (r undefined) are left out, as in training.
  run.sh, postpro.sh, clean.sh

Usage (from the repo root):
  python -m sweep.online CASE NET [TAG]     TAG defaults to NET
then  unsteady/online/<CASE>_<TAG>/run.sh 1  and its postpro.sh.
"""
import os
import shutil
import sys

import numpy as np

import common
from learn import dataset, pilot3d
from sweep import meshGen, newRun, oracle

onlineDir = os.path.join(common.UNSTEADY, 'online')


def setup(name, net, tag):
    d = os.path.join(onlineDir, f'{name}_{tag}')
    for sub in ('results', 'references', 'postpro'):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    for f in oracle.caseScripts:
        shutil.copy2(os.path.join(oracle.sweepDir, 'runTemplate', f), d)

    c = oracle.runCase(name)
    with open(oracle.templateFile) as fh:
        text = newRun.fillTemplate(fh.read(), oracle.substitutions(d, c),
                                   oracle.caseInfo(name, c))
    old = '    // Time series of the binned source per coarse node (format in sweep/oracle.py)\n' \
          '    OracleSourceFile = "oracleSource.txt";\n'
    weights = os.path.join(common.DATA, 'nn', net, 'nnWeights.txt')
    new = ('    // Neural source (sweep/online.py): network weights and column geometry\n'
           f'    NeuralSourceWeightsFile  = {newRun.quoted(weights, d)};\n'
           '    NeuralSourceGeometryFile = "nnGeometry.txt";\n')
    assert text.count(old) == 1 and os.path.isfile(weights)
    text = text.replace(old, new)
    header = f'// ONLINE run with the neural source {net} (sweep/online.py); otherwise the oracle setup:\n'
    with open(os.path.join(d, 'FluidFile'), 'w') as fh:
        fh.write(header + text)

    # columns the body touches: those of the oracle source
    src = np.load(os.path.join(oracle.oracleDir, 'runs', name, 'oracleSource.npz'))
    nodes, tets = common.readTop(oracle.meshFile)
    cols = np.array([src['xyz'][src['col'] == j][0, :2] for j in range(src['col'].max() + 1)])
    geo = meshGen.loadGeometry(os.path.join(common.UNSTEADY, 'runs', name))
    X = pilot3d.readSurfaceNodes(geo['surfFile'])
    xy = np.unique(np.round(X[:, :2], 12), axis=0)
    G = dataset.columnGeometry(geo, nodes, tets, xy, pilot3d.stationNormals(xy), cols)
    keep = np.isfinite(G['rx'])
    with open(os.path.join(d, 'nnGeometry.txt'), 'w') as fh:
        fh.write(f'{common.gamma} {common.R_gas} {common.C_suth} {common.S_suth}\n')
        for j in np.flatnonzero(keep):
            fh.write(' '.join(f'{v:.17g}' for v in (cols[j, 0], cols[j, 1], meshGen.h, geo['D'],
                     G['volFrac'][j], G['Nx'][j], G['Ny'][j], G['rx'][j], G['ry'][j])) + '\n')
    print(f'online/{name}_{tag}: net {net}, {keep.sum()} columns ({(~keep).sum()} sliver skipped)')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    setup(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else sys.argv[2])
