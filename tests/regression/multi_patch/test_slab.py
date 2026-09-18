import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mpi4py.MPI
import numpy as np
import pytest
import torch
from igakit.cad import refine, line

from ttnte import mpi_context
from ttnte.parallel import IGADofHeuristic
from ttnte.xs.benchmarks import pu239, research_reactor
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.mesh._gather import GatheredPatch, gather_mesh_patches
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
    IGADDSolver,
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
    dd_solver = IGADDSolver(driver.mesh, strategy)

    # DDSolver.set_callback()/TransportDriver.set_callback(): the callback is
    # handed the solver/driver itself, so it can pull whatever it needs off
    # their existing getters -- exercised here on a real, MPI-parallel,
    # multi-Schwarz-iteration DD eigenvalue solve (unlike the single-patch
    # fixed-source unit test, this actually runs several Schwarz sweeps per
    # outer iteration and several outer iterations).
    dd_calls = []
    dd_solver.set_callback(lambda solver: dd_calls.append(solver.eps))

    outer_calls = []

    def outer_callback(driver_arg, solver_arg):
        outer_calls.append(
            {
                "i": driver_arg.last_num_outer_iterations(),
                "k": driver_arg.last_k(),
                "eps": solver_arg.eps,
            }
        )

    driver.set_callback(outer_callback)

    # Run solver
    result = driver.solve_eigenvalue(dd_solver, tol=outer_tol, max_iter=100)
    k = result.k_eff
    assert 1e5 * abs(1 - k) < 20

    # AMEnSolver initializes eps_ directly to the constructor's `eps`
    # (== eps_floor_) BEFORE any forcing update has ever run, so the very
    # first callback observation -- captured before the first Schwarz
    # sweep's own update_convergence_criteria() call -- is that raw,
    # artificially tight floor value, not yet the forcing formula's own
    # output. From the SECOND observation onward, every value has gone
    # through eps_ = max(eps_floor_, eps_forcing_ * min_error_) with
    # min_error_ a running minimum (never increases -- see
    # AMEnSolver::update_convergence_criteria()), so that tail must be a
    # non-increasing sequence, across the WHOLE run (both Schwarz sweeps
    # within an outer iteration and across outer iterations). Every observed
    # eps is strictly positive and every observed patch state defined.
    assert len(dd_calls) > 0
    assert all(e > 0 for e in dd_calls)
    assert all(dd_calls[i] >= dd_calls[i + 1] for i in range(1, len(dd_calls) - 1))
    assert dd_solver.last_num_iterations() <= len(dd_calls)

    assert len(outer_calls) > 0
    assert all(c["eps"] > 0 for c in outer_calls)
    assert all(c["k"] is not None for c in outer_calls)
    assert [c["i"] for c in outer_calls] == list(range(1, len(outer_calls) + 1))
    assert driver.last_num_outer_iterations() == len(outer_calls)
    # The eigenvalue observed on the final outer iteration should already be
    # close to the fully-converged k_eff the solve settles on.
    assert 1e5 * abs(outer_calls[-1]["k"] - k) < 20

    # TransportSolution should hold exactly this rank's own local systems'
    # states, and compute_scalar_flux() should match the quadrature set's
    # own integrate() applied directly to those same states.
    scalar_result = result.compute_scalar_flux()
    for c in driver.mesh.blocks:
        gid = c.gid
        system_state = driver.get_system(gid).state

        torch.testing.assert_close(
            result.get_local_field(gid).to_dense(), system_state.to_dense()
        )
        torch.testing.assert_close(
            scalar_result.get_local_field(gid).to_dense(),
            qset.integrate(system_state).to_dense(),
        )

    # compute_errors() against itself (same discretization, same values)
    # should be ~0 for every local patch/group -- exercises both the
    # angular-integration path (on `result`, which still carries angular
    # dependence) and the spatial-only path (on `scalar_result`, post
    # compute_scalar_flux()). Local-only (no MPI): keyed by this rank's own
    # GIDs, matching driver.mesh.blocks exactly.
    self_errors = scalar_result.compute_errors(scalar_result)
    for c in driver.mesh.blocks:
        assert self_errors[c.gid].shape == (xs_server.num_groups,)
        assert torch.all(self_errors[c.gid] < 1e-4)

    angular_self_errors = result.compute_errors(result)
    for c in driver.mesh.blocks:
        assert angular_self_errors[c.gid].shape == (xs_server.num_groups,)
        assert torch.all(angular_self_errors[c.gid] < 1e-4)

    # Whole-mesh cross-rank compositing (gather_mesh_patches(), always
    # collective). Independently verify each gathered patch's field values
    # against a SEPARATE mpi4py pickled gather (not gather_mesh_patches()'s
    # own wire protocol) of what each rank computes locally for its own
    # patches -- a genuine cross-check of the binary send/recv protocol,
    # not just "didn't crash".
    plot_resolution = 6
    local_expected = {}
    for c in driver.mesh.blocks:
        gid = c.gid
        ctrlpts_shape = [c.get_ctrlpts_size(d) for d in range(c.ndim)]
        field = (
            scalar_result.get_local_field(gid).to_dense().reshape(*ctrlpts_shape, -1)
        )
        tensor_product_pts = c.ndim * [
            torch.linspace(0, 1, plot_resolution, dtype=dtype)
        ]
        mesh_coords = torch.meshgrid(tensor_product_pts, indexing="ij")
        flat_points = torch.stack([m.flatten() for m in mesh_coords], dim=-1)
        values = c.evaluate_field(field, flat_points)[..., 0].reshape(
            c.ndim * [plot_resolution]
        )
        local_expected[gid] = values.to(torch.float64).cpu().numpy()

    all_expected = mpi4py.MPI.COMM_WORLD.gather(local_expected, root=0)

    gathered = gather_mesh_patches(
        driver.mesh,
        resolution=plot_resolution,
        solution=scalar_result,
        root=0,
    )
    if mpi_context.rank == 0:
        assert all_expected is not None
        combined_expected = {}
        for d in all_expected:
            combined_expected.update(d)

        expected_gids = sorted(driver.mesh.gid2rank.keys())
        assert len(gathered) == len(expected_gids)
        for gid, patch in zip(expected_gids, gathered):
            if isinstance(patch, GatheredPatch):
                assert patch.has_field
                actual = patch.field_values.to(torch.float64).cpu().numpy()
            else:
                assert patch.gid == gid
                actual = local_expected[gid]
            np.testing.assert_allclose(
                actual, combined_expected[gid], atol=1e-5, rtol=1e-5
            )
    else:
        assert gathered == []

    # Smoke-test the collective, whole-mesh composited plot built on top of
    # gather_mesh_patches() -- must not raise on any rank, and must follow
    # the documented per-rank return convention (Axes on root, None
    # elsewhere).
    ax = driver.mesh.plot(
        resolution=plot_resolution,
        solution=scalar_result,
        gather=True,
        backend="matplotlib",
    )
    if mpi_context.rank == 0:
        assert ax is not None
        plt.close(ax.figure)
    else:
        assert ax is None

    # error_norm() is the whole-solution aggregate (collective across every
    # rank, per energy group) -- also ~0 when compared against itself.
    self_norm = scalar_result.error_norm(scalar_result)
    assert self_norm.shape == (xs_server.num_groups,)
    assert torch.all(self_norm < 1e-4)

    angular_self_norm = result.error_norm(result)
    assert angular_self_norm.shape == (xs_server.num_groups,)
    assert torch.all(angular_self_norm < 1e-4)

    # regular_mesh_average() -- volume-averaged scalar flux on a regular
    # 1-D grid via composite trapezoidal integration; collective across
    # ranks. The slab (c0, c1 mirrored about x=0) should give a symmetric
    # averaged flux.
    avg = scalar_result.regular_mesh_average([8], [5])
    assert avg.shape == (8, xs_server.num_groups)
    assert torch.all(avg > 0)
    torch.testing.assert_close(avg, avg.flip(0), atol=1e-3, rtol=1e-3)


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
    dd_solver = IGADDSolver(driver.mesh, strategy)

    # Run solver
    result = driver.solve_eigenvalue(dd_solver, tol=outer_tol, max_iter=100)
    k = result.k_eff
    assert 1e5 * abs(1 - k) < 20

    # select_group() should narrow the energy axis (size num_groups, this
    # benchmark's only multigroup case in this file) down to size 1, matching
    # a direct dense-tensor slice -- exercised on both the still-angular
    # result and the spatial-only scalar flux.
    scalar_result = result.compute_scalar_flux()
    for c in driver.mesh.blocks:
        gid = c.gid
        dense = scalar_result.get_local_field(gid).to_dense()
        assert dense.shape[1] == xs_server.num_groups
        for g in range(xs_server.num_groups):
            single = scalar_result.select_group(g)
            single_dense = single.get_local_field(gid).to_dense()
            torch.testing.assert_close(single_dense, dense.narrow(1, g, 1))
            # k_eff/gid2rank carry over unchanged.
            assert single.k_eff == scalar_result.k_eff

        # Out-of-range group must raise, not silently misbehave.
        with pytest.raises(RuntimeError):
            scalar_result.select_group(xs_server.num_groups)

        # select_group() also works directly on the still-angular result.
        angular_single = result.select_group(0)
        assert angular_single.k_eff == result.k_eff

        # Plotting a raw multigroup field must raise, not silently render
        # group 0 -- narrow via select_group() first.
        with pytest.raises(ValueError):
            c.plot(field=dense, backend="matplotlib", resolution=6)

        single_dense = scalar_result.select_group(0).get_local_field(gid).to_dense()
        ax = c.plot(field=single_dense, backend="matplotlib", resolution=6)
        assert ax is not None
        plt.close(ax.figure)
