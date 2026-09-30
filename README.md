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
10. Your output: EmbeddedSurfaceForce copies Fs, reduces it to rank 0 (the per-node equivalent of step 9), scales it to newtons, and writes one "Fx Fy Fz" line per structure node. The sum of the lines equals LiftandDrag, which the Phase 3 check confirmed.

For the source term: cell K's force on the fluid is −Σ_{a∈K} Fs[a]. Add back p∞ · (1/3) Σ A n_out for each node to get absolute pressure, and divide by span dz for a per-unit-span value.