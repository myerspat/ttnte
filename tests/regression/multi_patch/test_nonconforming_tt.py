"""3-D non-conforming interface: block-Jacobi AMEn against the sweeper.

Two cylindrical rings joined on r = 2, refined differently on the two sides, the
outer ring with its parametric axes reordered so its face parameters are transposed
with respect to the inner ring's (the TT inflow operator then swaps the neighbor's
basis cores)."""

import math

import pytest
import torch
from igakit import cad

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.driver import IGATransportDriver3D
from ttnte.math import ProductQuadrature
from ttnte.mesh import IGAMesh
from ttnte.physics import DGTransportAssemblerConfig, FixedSource
from ttnte.solvers import (
    AMEnSolver,
    BlockJacobiStrategy,
    CommMode,
    DDSolverConfig,
    ExecMode,
    IGADDSolver,
    MemoryPolicy,
)
from ttnte.sweep import IGASweepDriver3D, SweepConfig
from ttnte.xs import Material, Server

CPU = torch.device("cpu")


def _xs():
    server = Server()
    fills = []
    for name, total, scatter in [("Inner", 1.0, 0.5), ("Outer", 0.6, 0.3)]:
        m = Material(name)
        m.total = torch.tensor([total])
        m.scatter_gtg = torch.tensor([[[scatter]]])
        m.finalize()
        server.add_material(m)
        fills.append(m)
    server.finalize()
    return fills, server


def _mesh(fills):
    inner = cad.ruled(
        cad.circle(radius=1, angle=math.pi / 2), cad.circle(radius=2, angle=math.pi / 2)
    )
    outer = cad.ruled(
        cad.circle(radius=2, angle=math.pi / 2), cad.circle(radius=3, angle=math.pi / 2)
    )
    inner = cad.refine(cad.extrude(inner, 1.0, 2), factor=(2, 2, 2), degree=2)
    # (angle, r, z) -> (z, angle, r), 3 angular and 3 axial spans
    outer = cad.refine(cad.extrude(outer, 1.0, 2), factor=(3, 2, 3), degree=2)
    outer = outer.transpose((2, 0, 1))
    mesh = IGAMesh(mpi_context)
    for k, nrb in enumerate([inner, outer]):
        patch = Patch.from_igakit(
            nrb, device=CPU, dtype=torch.float64, fill=fills[k].label
        )
        if k == 0:
            patch.source = FixedSource(isotropic_strength=torch.tensor([1.0]))
        mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()
    return mesh


def _qset():
    qset = ProductQuadrature.gauss_legendre_chebyshev(2, 2, 3)
    qset.to_(CPU, torch.float64)
    return qset


@pytest.mark.mpi(min_size=1)
def test_nonconforming_3d_amen_matches_sweep():
    if mpi_context.world_size > 1:
        pytest.skip("Sweeps run on one rank")
    torch.set_default_dtype(torch.float64)
    fills, xs = _xs()

    mesh = _mesh(fills)
    conns = [
        c
        for b in mesh.blocks
        for d in range(3)
        for up in (False, True)
        for c in b.get_boundary_info(d, up).connections
    ]
    assert len(conns) == 2 and not any(c.conforming for c in conns)
    assert any(list(c.mapping.perm) != [0, 1, 2] for c in conns)

    sweeper = IGASweepDriver3D(mesh, xs, mpi_context)
    sweeper.setup(_qset(), SweepConfig(tol=1e-11, max_iter=3000, verbose=False))
    sweeper.solve_fixed_source()
    assert sweeper.converged

    cfg = DGTransportAssemblerConfig()
    cfg.rounding.eps = 1e-13
    cfg.cross.eps = 1e-13
    cfg.max_dense_size = int(1e10)
    cfg.cross_jacobian_inverse = False
    driver = IGATransportDriver3D(_mesh(fills), xs, mpi_context)
    qset = _qset()
    driver.assemble(qset, cfg)
    strategy = BlockJacobiStrategy(
        DDSolverConfig(
            tol=1e-11,
            max_iter=5000,
            use_gpu=False,
            memory_policy=MemoryPolicy.RESIDENT,
            exec_mode=ExecMode.ASYNC,
            comm_mode=CommMode.ASYNC,
            verbose=False,
        )
    )
    strategy.set_local_solver(
        AMEnSolver(nswp=4, eps=1e-12, kickrank=4, local_iterations=200, resets=4)
    )
    driver.solve_fixed_source(
        IGADDSolver(driver.mesh, strategy), tol=1e-10, max_iter=300, verbose=False
    )
    qset.to_(CPU, torch.float64)

    num = den = 0.0
    for gid in driver.gid2rank:
        ref = qset.integrate(driver.get_system(gid).state, 0.0).as_tt().to_dense()
        phi = sweeper.get_scalar_flux(gid).reshape(ref.shape)
        num += ((phi - ref) ** 2).sum().item()
        den += (ref**2).sum().item()
    assert (num / den) ** 0.5 < 1e-8
