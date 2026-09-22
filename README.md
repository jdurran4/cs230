# CS230 — Learned Source Terms for Unresolved Immersed Bodies

CS230 Deep Learning final project and initial proof-of-concept for a data-driven
framework for accelerating CFD simulations containing small immersed bodies.

## Overview

Small immersed bodies such as cables can require substantial mesh refinement in
high-fidelity CFD simulations. The goal of this project is to investigate whether
the effect of a geometrically resolved body can instead be represented by a
learned equivalent source term applied to a coarse CFD mesh.

The initial test problem is unsteady quasi-2D compressible flow over a stationary
cylinder using AERO-F.

A coarse background mesh defines the control volumes that will eventually be used
for unresolved simulations. The cylinder is resolved using embedded-boundary
adaptive mesh refinement (AMR). Resolved flow quantities and surface forces are
then filtered onto the original coarse control volumes to construct a supervised
learning dataset.

For each coarse cell intersecting the cylinder, the initial model will use inputs

- local Mach number
- local Reynolds number
- solid volume fraction
- relative flow/geometry angle

and predict the equivalent momentum source terms

- x-momentum source
- y-momentum source

The source terms are obtained by integrating the pressure and viscous traction
over the portion of the cylinder contained within each coarse control volume.

## Workflow

1. Generate a parameterized quasi-2D embedded-cylinder CFD problem.
2. Pre-refine the cylinder using AMR and run an unsteady resolved simulation.
3. Discard the initial transient and collect flow snapshots.
4. Average resolved flow quantities over the original coarse control volumes.
5. Integrate cylinder surface forces within each coarse control volume.
6. Construct the machine-learning dataset.
7. Train and evaluate regression / neural-network models.
8. Test generalization across Mach number, Reynolds number, and cylinder position.

## Long-Term Goal

This cylinder problem is a simplified demonstration of a broader framework for
learning equivalent conservation-law source terms for unresolved immersed bodies.

The eventual application is accelerating fluid-structure interaction simulations
of supersonic parachutes by replacing geometrically resolved suspension lines
with learned source terms on the coarse CFD mesh.