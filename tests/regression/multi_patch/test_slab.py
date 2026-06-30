import pytest
import torch
from igakit.cad import refine, line

from ttnte import mpi_context
from ttnte.parallel import IGADofHeuristic
from ttnte.xs.benchmarks import pu239, research_reactor
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
)
from ttnte.math import QuadratureSet1D
from ttnte.linalg import Operator, TTEngine, mm
from ttnte.driver import IGATransportDriver1D
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
def test_homogeneous_slab(device, memory_policy, dtype):
    """"""
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Initialize the context
    mpi_context.init()

    # Check MPI size
    if mpi_context.world_size > 2:
        pytest.skip("Test requires exactly 2 processes")

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # Get XS info
    fills, xs_server = pu239(num_groups=1, device=torch.device("cpu"), dtype=dtype)
    assert fills[0].to_string() == "Pu-239"
    assert xs_server.num_groups == 1

    # Create single-patch geometry (homogeneous circle)
    rc = 2.256751
    c0 = Patch.from_igakit(
        refine(line((-rc, 0), (0, 0)), 5, 3),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fills[0],
    )
    c1 = Patch.from_igakit(
        refine(line((0, 0), (rc, 0)), 5, 3),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fills[0],
    )

    assert c0.is_finalized() and c1.is_finalized()
    assert not c0.is_rational() and not c1.is_rational()
    assert c0.ndim == 1 and c1.ndim == 1
    assert c0.ctrlptsw.shape == (8, 1) and c1.ctrlptsw.shape == (8, 1)
    assert c0.device == torch.device("cpu") and c1.device == torch.device("cpu")
    assert c0.dtype == dtype and c1.dtype == dtype
    assert c0.get_numel(0) == 5 and c1.get_numel(0) == 5
    assert c0.degrees[0] == 3 and c1.degrees[0] == 3

    mesh = IGAMesh(mpi_context)
    mesh.add_block(c0)
    mesh.add_block(c1)
    assert not mesh.is_finalized() and not mesh.is_connected()
    assert mesh.num_blocks == 2
    mesh.connect()
    mesh.finalize()
    assert mesh.is_finalized() and mesh.is_connected()

    # After connect+finalize: shared face becomes INTERNAL, free ends become VACUUM
    assert c0.get_boundary_info(0, False).type == BoundaryType.VACUUM
    assert c0.get_boundary_info(0, True).type == BoundaryType.INTERNAL
    assert c1.get_boundary_info(0, False).type == BoundaryType.INTERNAL
    assert c1.get_boundary_info(0, True).type == BoundaryType.VACUUM

    # Create 1-D angular quadrature
    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(torch.device("cpu"), dtype)

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6 if dtype == torch.float32 else 1e-5
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else 0
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True

    # Create the transport driver
    driver = IGATransportDriver1D(mesh, xs_server, mpi_context)

    # Distribute patches among MPI ranks
    if mpi_context.world_size != 1:
        driver.distribute([IGADofHeuristic()])

    # Run the assembler
    driver.assemble(qset, config)

    for c in driver.mesh.blocks:
        # Get the assembler
        assembler = driver.get_assembler(c.gid)

        # Get operators from the assembler
        H = assembler.interior_loss_op
        S = assembler.scatter_op
        F = assembler.fission_op
        Bin = assembler.inflow_ops
        Bout = assembler.outflow_ops

        assert len(Bin) == 2
        assert len(Bout) == 2

        # Check the operators (not inflow)
        for op in [H, S, F] + Bout:
            assert isinstance(op, Operator) and op.is_tt

            en = op.as_tt()
            assert len(en) == 3
            assert en.device == torch.device("cpu")
            assert en.dtype == dtype
            assert en[0].shape[:-1] == (1, 64, 64)
            assert en[1].shape[1:-1] == (
                c.get_numel(0) + c.degrees[0],
                c.get_numel(0) + c.degrees[0],
            )
            assert en[2].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)

            for i in range(len(en) - 1):
                assert en[i].shape[-1] == en[i + 1].shape[0]

        # Check the inflow operators
        for op in Bin:
            if not op.defined():
                continue

            assert isinstance(op, Operator) and op.is_tt
            en = op.as_tt()
            assert len(en) == 3
            assert en.device == torch.device("cpu")
            assert en.dtype == dtype
            assert en[0].shape[:-1] == (1, 64, 64)
            assert en[1].shape[1:-1] == (c.get_numel(0) + c.degrees[0], 1)
            assert en[2].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)

            for i in range(len(en) - 1):
                assert en[i].shape[-1] == en[i + 1].shape[0]

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
        max_iter=10,
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


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("device, memory_policy, dtype", test_params)
def test_research_reactor(device, memory_policy, dtype):
    """"""
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Initialize the context
    mpi_context.init()

    # Check MPI size
    if mpi_context.world_size > 2:
        pytest.skip("Test requires exactly 2 processes")

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # Get XS info
    fills, xs_server = research_reactor(
        is_anisotropic=False, dtype=dtype, device=torch.device("cpu")
    )
    assert fills[0].to_string() == "Fuel"
    assert fills[1].to_string() == "Moderator"
    assert xs_server.num_groups == 2

    # Create single-patch geometry (homogeneous circle)
    c0 = Patch.from_igakit(
        refine(line((-6.696802, 0), (0, 0)), 5, 3),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fills[0],
    )
    c1 = Patch.from_igakit(
        refine(line((0, 0), (1.126152, 0)), 5, 3),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fills[1],
    )

    assert c0.is_finalized() and c1.is_finalized()
    assert not c0.is_rational() and not c1.is_rational()
    assert c0.ndim == 1 and c1.ndim == 1
    assert c0.ctrlptsw.shape == (8, 1) and c1.ctrlptsw.shape == (8, 1)
    assert c0.device == torch.device("cpu") and c1.device == torch.device("cpu")
    assert c0.dtype == dtype and c1.dtype == dtype
    assert c0.get_numel(0) == 5 and c1.get_numel(0) == 5
    assert c0.degrees[0] == 3 and c1.degrees[0] == 3

    mesh = IGAMesh(mpi_context)
    mesh.add_block(c0)
    mesh.add_block(c1)
    assert not mesh.is_finalized() and not mesh.is_connected()
    assert mesh.num_blocks == 2
    mesh.connect()
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True), BoundaryType.REFLECTIVE, tol=1e-6
    )
    mesh.finalize()
    assert mesh.is_finalized() and mesh.is_connected()

    # After connect+finalize: shared face becomes INTERNAL, free ends become VACUUM
    assert c0.get_boundary_info(0, False).type == BoundaryType.REFLECTIVE
    assert c0.get_boundary_info(0, True).type == BoundaryType.INTERNAL
    assert c1.get_boundary_info(0, False).type == BoundaryType.INTERNAL
    assert c1.get_boundary_info(0, True).type == BoundaryType.VACUUM

    # Create 1-D angular quadrature
    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(torch.device("cpu"), dtype)

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6 if dtype == torch.float32 else 1e-5
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else 0
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True

    # Create the transport driver
    driver = IGATransportDriver1D(mesh, xs_server, mpi_context)

    # Distribute patches among MPI ranks
    if mpi_context.world_size != 1:
        driver.distribute([IGADofHeuristic()])

    # Run the assembler
    driver.assemble(qset, config)

    for c in driver.mesh.blocks:
        # Get the assembler
        assembler = driver.get_assembler(c.gid)

        # Get operators from the assembler
        H = assembler.interior_loss_op
        S = assembler.scatter_op
        F = assembler.fission_op
        Bin = assembler.inflow_ops
        Bout = assembler.outflow_ops

        assert len(Bin) == 2
        assert len(Bout) == 2

        # Check the operators (not inflow)
        for op in [H, S, F] + Bout:
            if not op.defined():
                continue

            assert isinstance(op, Operator) and op.is_tt

            en = op.as_tt()
            assert len(en) == 3
            assert en.device == torch.device("cpu")
            assert en.dtype == dtype
            assert en[0].shape[:-1] == (1, 64, 64)
            assert en[1].shape[1:-1] == (
                c.get_numel(0) + c.degrees[0],
                c.get_numel(0) + c.degrees[0],
            )
            assert en[2].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)

            for j in range(len(en) - 1):
                assert en[j].shape[-1] == en[j + 1].shape[0]

        # Check the inflow operators
        for op in Bin:
            if not op.defined():
                continue

            assert isinstance(op, Operator) and op.is_tt
            en = op.as_tt()
            assert len(en) == 3
            assert en.device == torch.device("cpu")
            assert en.dtype == dtype
            assert en[0].shape[:-1] == (1, 64, 64)
            assert en[1].shape[1:-1] == (c.get_numel(0) + c.degrees[0], 1) or en[
                1
            ].shape[1:-1] == (
                c.get_numel(0) + c.degrees[0],
                c.get_numel(0) + c.degrees[0],
            )
            assert en[2].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)

            for j in range(len(en) - 1):
                assert en[j].shape[-1] == en[j + 1].shape[0]

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
        max_iter=10,
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
