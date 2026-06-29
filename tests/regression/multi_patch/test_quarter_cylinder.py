import pytest
import torch
import numpy as np
from igakit.cad import circle, refine, line, coons, ruled, bilinear

from ttnte import mpi_context
from ttnte.parallel import IGADofHeuristic
from ttnte.xs.benchmarks import pu239
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
)
from ttnte.math import ProductQuadrature
from ttnte.linalg import Operator, TTEngine, mm
from ttnte.driver import IGATransportDriver2D
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    ExecMode,
    CommMode,
)

test_params = [
    ("cpu", MemoryPolicy.OUT_OF_CORE, torch.float32),
    ("cuda", MemoryPolicy.OUT_OF_CORE, torch.float32),
    ("cuda", MemoryPolicy.RESIDENT, torch.float32),
    ("cuda", MemoryPolicy.STATE_RESIDENT, torch.float32),
    ("cuda", MemoryPolicy.OPERATOR_RESIDENT, torch.float32),
    ("cpu", MemoryPolicy.OUT_OF_CORE, torch.float64),
    ("cuda", MemoryPolicy.OUT_OF_CORE, torch.float64),
    ("cuda", MemoryPolicy.RESIDENT, torch.float64),
    ("cuda", MemoryPolicy.STATE_RESIDENT, torch.float64),
    ("cuda", MemoryPolicy.OPERATOR_RESIDENT, torch.float64),
]


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("device, memory_policy, dtype", test_params)
def test_reflected_pu235_cylinder(device, memory_policy, dtype):
    """"""
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Initialize the context
    mpi_context.init()

    # Check MPI size
    if mpi_context.world_size > 3:
        pytest.skip("Test requires 3 or fewer processes")

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # Get XS info
    fills, xs_server = pu239(num_groups=1, device=torch.device("cpu"), dtype=dtype)
    assert fills[0].to_string() == "Pu-239"
    assert fills[1].to_string() == "Water"
    assert xs_server.num_groups == 1

    # Create quarter cylinder of Pu-235
    rc0 = 3.397610
    c0 = circle(rc0, angle=torch.pi / 4)
    c1 = circle(rc0, angle=-torch.pi / 4).rotate(torch.pi / 2)
    l0 = line((0, 0), (rc0, 0))
    l1 = line((0, 0), (0, rc0))

    s0 = coons([[l1, c0], [l0, c1]])
    c0 = s0.boundary(0, 1)
    c1 = s0.boundary(1, 1)

    s0 = Patch.from_igakit(
        refine(s0, 6, 3),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fills[0],
    )

    # Create two patch annular water reflector
    rc1 = 3.063725
    c2 = circle(rc0 + rc1, angle=torch.pi / 4)
    c3 = circle(rc0 + rc1, angle=-torch.pi / 4).rotate(torch.pi / 2)
    s1 = Patch.from_igakit(
        refine(ruled(c0, c2), 6, 3),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fills[1],
    )
    s2 = Patch.from_igakit(
        refine(ruled(c1, c3), 6, 3),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fills[1],
    )

    for s in [s0, s1, s2]:
        assert s.is_finalized()
        assert s.ndim == 2
        assert s.ctrlptsw.shape == (9, 9, 3)
        assert s.device == torch.device("cpu")
        assert s.dtype == dtype
        assert s.get_numel(0) == 6
        assert s.get_numel(1) == 6
        assert s.degrees[0] == 3 and s.degrees[1] == 3

    mesh = IGAMesh(mpi_context)
    mesh.add_block(s0)
    mesh.add_block(s1)
    mesh.add_block(s2)
    assert not mesh.is_finalized() and not mesh.is_connected()
    assert mesh.num_blocks == 3

    mesh.connect()
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True),
        BoundaryType.REFLECTIVE,
        tol=1e-6,
    )
    mesh.finalize()

    # Check boundary conditions
    assert s0.get_boundary_info(0, False).type == BoundaryType.REFLECTIVE
    assert s0.get_boundary_info(0, True).type == BoundaryType.INTERNAL
    assert s0.get_boundary_info(1, False).type == BoundaryType.REFLECTIVE
    assert s0.get_boundary_info(1, True).type == BoundaryType.INTERNAL

    assert s1.get_boundary_info(0, False).type == BoundaryType.REFLECTIVE
    assert s1.get_boundary_info(0, True).type == BoundaryType.INTERNAL
    assert s1.get_boundary_info(1, False).type == BoundaryType.INTERNAL
    assert s1.get_boundary_info(1, True).type == BoundaryType.VACUUM

    assert s2.get_boundary_info(0, False).type == BoundaryType.REFLECTIVE
    assert s2.get_boundary_info(0, True).type == BoundaryType.INTERNAL
    assert s2.get_boundary_info(1, False).type == BoundaryType.INTERNAL
    assert s2.get_boundary_info(1, True).type == BoundaryType.VACUUM
    assert mesh.is_finalized() and mesh.is_connected()

    # Create angular quadrature
    qset = ProductQuadrature.gauss_legendre_chebyshev(8, 8, s0.ndim)
    qset.to_(torch.device("cpu"), dtype)

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6 if dtype == torch.float64 else 1e-5
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else 0
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True

    # Create the transport driver
    driver = IGATransportDriver2D(mesh, xs_server, mpi_context)

    # Distribute patches among MPI ranks
    if mpi_context.world_size != 1:
        driver.distribute([IGADofHeuristic()])

    # Run the assembler
    driver.assemble(qset, config)

    outer_tol = 1e-6
    inner_tol = 1e-7
    eps = 1e-8
    if dtype == torch.float32:
        outer_tol = 1e-4
        inner_tol = 1e-5
        eps = 1e-6

    # Create Block-Jacobi DD strategy
    config = DDSolverConfig(
        tol=inner_tol,
        max_iter=50,
        eps=eps,
        use_gpu=True if device == "cuda" else False,
        memory_policy=memory_policy,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=True,
    )
    strategy = BlockJacobiStrategy(config)
    strategy.set_local_solver(
        AMEnSolver(nswp=2, eps=eps, kickrank=2, local_iterations=60, resets=4)
    )

    # Run solver
    k = driver.solve_eigenvalue(strategy, tol=outer_tol, max_iter=100)
    assert 1e5 * abs(1 - k) < 20
