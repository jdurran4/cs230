# CS230 — Learned Source Terms for Unresolved Immersed Bodies

CS230 Deep Learning final project and initial proof-of-concept for a data-driven
framework for accelerating CFD simulations containing small immersed bodies.

## Overview

Small immersed bodies such as cables can require substantial mesh refinement in
high-fidelity CFD simulations. The goal of this project is to investigate whether
the effect of a geometrically resolved body can instead be represented by a
learned equivalent source term applied to a coarse CFD mesh.

The test problem is quasi-2D compressible flow over a stationary cylinder using AERO-F.
A coarse background mesh defines the control volumes (dual cells of its nodes) used by
unresolved simulations. The cylinder is resolved with embedded-boundary adaptive mesh
refinement (AMR). The resolved surface force is integrated over the part of the cylinder
inside each coarse control volume, and the resolved flow is averaged over the same
volumes, giving a supervised dataset: per coarse column, flow state + geometry → source.

## Long-Term Goal

This cylinder problem is a simplified demonstration of a broader framework for
learning equivalent conservation-law source terms for unresolved immersed bodies.

The eventual application is accelerating fluid-structure interaction simulations
of supersonic parachutes by replacing geometrically resolved suspension lines
with learned source terms on the coarse CFD mesh. There, robustness matters more than
accuracy: the source must stay physical under chaotic, widely varying states.

## Status

**steady-v0** (git tag in this repo and in AERO-F `source-terms`) is the first complete
pass: steady resolved sweeps (M 1.2–2.5, Re_D 20–200, random cylinder offsets, multiple
coarse grids D/h 0.25–2.4 in post-processing), a flow-frame MLP, and AERO-F source terms
(oracle from file, and the NN online). Lessons that drive the next version:

- An oracle (true source read from a file) on the coarse mesh reproduces the far-field
  wake well; near-field waves are weaker than resolved.
- Training inputs must come from the state the coarse solver actually sees: a net
  trained on coarse oracle states was ~9× more accurate on them than one trained on
  fine-solution averages.
- Online, from a freestream start, the coarse-state net predicts thrust, no bow shock
  forms, and the run converges to a wrong state (Cd 0.11 vs 1.76). Restarted from the
  oracle solution it holds Cd 1.63. Training data must cover the path, not only the answer.
- The flow-aligned frame (parallel/perpendicular to the local velocity) is ill-defined
  where the local flow stagnates (the column holding most of the body), and the
  prediction there flips sign.
- In coarse body columns the node velocity is largely an effect of the source itself, so
  a local drag law is structurally wrong there (the true force pushes along the local
  coarse flow in 66% of rows).

## Next design

- Inputs and outputs in the solver's Cartesian x-y frame: local Mach vector, Re_c =
  ρcD/μ (instead of Re_D, which is singular at zero velocity), volume fraction, area
  normal N, centroid offset r/h, D/h; outputs Sx, Sy (minus the pressure baseline p̄DN,
  normalized by (p̄+q̄)D). Rotation invariance, when needed, by data augmentation.
- Nodal state gradients as inputs: neighbor information for free, since AERO-F already
  computes them for second-order fluxes (first-order runs simply omit them).
- Static closure (no history inputs). Later: velocity relative to a moving cable and the
  matching energy source.
- Unsteady resolved runs with snapshots for training coverage, after a review of solver
  settings and physics.

## Layout

Run everything from the repo root as modules (`python -m learn.train cv`).

- `common.py`: data paths, gas constants, AERO-F file readers.
- `learn/dataset.py`: coarse-state dataset from the steady-v0 coarse oracle runs (raw SI,
  x-y frame, nodal least-squares gradients as AERO-F computes them).
- `learn/features.py`: input sets (steady-v0 flow frame, x-y, gradients) and targets.
- `learn/train.py`: cross-validated comparison of input sets, and the weights export for
  AERO-F.
- `online/check.py`: Python reference of AERO-F's neural source (test values for the C++).
- Data lives outside the repo in `~/Research/cs230-data/` (override with `CS230_DATA`):
  `steady-v0/` holds everything from the steady pass in its old layout (runs, coarse
  oracle/online runs, meshes, old datasets and weights, plus a copy of `tools.sh` so old
  run directories still run); `datasets/` and `nn/` hold what the new code writes. The
  steady-v0 scripts (case generation, binning, oracle setup) are in the `steady-v0` tag
  and get ported here when needed.

## How AERO-F computes the force on an embedded surface (default CLOSEST_ELEMENT algorithm):

1. Loop over the structure's surface triangles. Each triangle's area-weighted normal (½ of the cross product of two edges) gives n·dA in one vector.
2. Place a few quadrature points on each triangle (4 by default). The force integral over the triangle will be approximated as a weighted sum over these points.
3. Find the fluid tet that contains each quadrature point. Each rank searches only its own subdomains and skips points it doesn't own, so each rank ends up with a partial sum.
4. Pick the nearest real fluid node on each side of the surface. Ghost or inactive nodes inside the body are skipped, so a solid cylinder only gets a contribution from the outside.
5. Extrapolate the flow state to the quadrature point: V_wall = V_node + ∇V · (x_q − x_node). This uses the limited nodal gradients.
6. Compute the traction at that point:
   - pressure: (p − p∞) · n, with n pointing from the fluid into the body
   - wall shear, from velocity gradients using the ghost states (viscous runs only).

   This is the force on the body.
7. Split each point's force among the triangle's 3 nodes. Each node's fraction is the point's barycentric coordinate for that node, i.e. the linear "tent" shape function. The result is Fs[a] = ∫ φ_a · t dA. The fractions sum to 1, so no force is lost.
8. Total per rank: Steps 1–7 run for each subdomain, starting from zero at every call. The subdomain results are added into this rank's Fs, which holds only its own share.
9. Total force: getForcesAndMoments adds Fs[a] over all nodes, and globalSum adds up the ranks. Multiplying by ρU²L² converts to newtons, and rotating by α, β gives LiftandDrag. All of it, including the shear, goes into the "inviscid" bucket, which is why ViscousLiftandDrag is ≈ 0 for embedded surfaces.
10. Your output: EmbeddedSurfaceForce copies Fs, reduces it to rank 0 (the per-node equivalent of step 9), scales it to newtons, and writes one "Fx Fy Fz" line per structure node. The sum of the lines equals LiftandDrag, which the Phase 3 check confirmed.

For the source term: cell K's force on the fluid is −Σ_{a∈K} Fs[a]. Add back p∞ · (1/3) Σ A n_out for each node to get absolute pressure, and divide by span dz for a per-unit-span value.
