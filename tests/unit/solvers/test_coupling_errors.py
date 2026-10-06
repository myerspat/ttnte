"""Per-coupling Schwarz error exposed on NeighborCoupling.

`NeighborCoupling.sq_diff` / `sq_prev` (0.5 * squared L2 norms of the change in,
and previous value of, the outgoing partial current on each INTERNAL face, written
by LocalSolver.postsolve()) must be exactly the per-face pieces the DDSolver sums
into its global Schwarz error: sqrt(sum sq_diff / sum sq_prev) == last_errors()[-1]
after every sweep.
"""

import math

import torch
from igakit.cad import line, refine

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.driver import IGATransportDriver1D
from ttnte.math import QuadratureSet1D
from ttnte.mesh import IGAMesh
from ttnte.physics import DGTransportAssemblerConfig
from ttnte.solvers import (
    AMEnSolver,
    BlockJacobiStrategy,
    CommMode,
    DDSolverConfig,
    ExecMode,
    IGADDSolver,
    MemoryPolicy,
)
from ttnte.xs.benchmarks import pu239

DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _three_patch_slab():
    mpi_context.init()
    torch.set_default_dtype(DTYPE)
    torch.autograd.set_grad_enabled(False)
    fills, xs_server = pu239(num_groups=1, device=DEVICE, dtype=DTYPE)
    rc = 2.256751
    bounds = [-rc, -rc / 3, rc / 3, rc]
    mesh = IGAMesh(mpi_context)
    for a, b in zip(bounds[:-1], bounds[1:]):
        mesh.add_block(
            Patch.from_igakit(
                refine(line((a, 0), (b, 0)), 4, 3),
                device=DEVICE,
                dtype=DTYPE,
                fill=fills[0],
            )
        )
    mesh.connect()
    mesh.finalize()

    qset = QuadratureSet1D.gauss_legendre(16)
    qset.to_(DEVICE, DTYPE)
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-8
    config.cross.eps = config.rounding.eps
    config.max_dense_size = 0
    config.cross_jacobian_inverse = True
    driver = IGATransportDriver1D(mesh, xs_server, mpi_context)
    driver.assemble(qset, config)
    return driver


def test_coupling_sq_terms_reproduce_dd_error():
    driver = _three_patch_slab()
    dd_config = DDSolverConfig(
        tol=1e-7,
        max_iter=8,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(dd_config)
    strategy.set_local_solver(
        AMEnSolver(nswp=2, eps=1e-8, kickrank=2, local_iterations=60, resets=4)
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)

    checks = []

    def callback(solver):
        sq_diff = sq_prev = 0.0
        num_couplings = 0
        for sys in solver.local_systems:
            for c in sys.couplings:
                assert c.sq_diff >= 0.0 and c.sq_prev >= 0.0
                assert c.current_op.defined()
                sq_diff += c.sq_diff
                sq_prev += c.sq_prev
                num_couplings += 1
        assert num_couplings == 4  # two interfaces, seen from both sides
        mine = math.sqrt(sq_diff / sq_prev) if sq_prev > 0.0 else 0.0
        checks.append((mine, solver.last_errors()[-1]))

    dd_solver.set_callback(callback)
    driver.solve_eigenvalue(dd_solver, tol=1e-5, max_iter=6, verbose=False)

    assert len(checks) > 3
    assert any(theirs > 0.0 for _, theirs in checks)
    for mine, theirs in checks:
        assert abs(mine - theirs) <= 1e-12 * max(abs(theirs), 1e-300)
