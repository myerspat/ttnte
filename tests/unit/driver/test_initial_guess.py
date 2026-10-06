"""TransportDriver `initial_guess=` on solve_eigenvalue()/solve_fixed_source().

The contract under test: a per-GID starting angular flux replaces the default
all-ones guess; a converged solution fed back in as the guess re-converges
almost immediately to the same answer; a partial guess falls back to ones for
the patches it omits; and a guess that cannot belong to this problem is
rejected with an error instead of silently producing garbage.
"""

import pytest
import torch
from igakit.cad import line, refine

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.math import QuadratureSet1D
from ttnte.mesh import IGAMesh
from ttnte.physics import BoundaryType, DGTransportAssemblerConfig, FixedSource
from ttnte.solvers import (
    AMEnSolver,
    BlockJacobiStrategy,
    CommMode,
    DDSolverConfig,
    ExecMode,
    IGADDSolver,
    MemoryPolicy,
)
from ttnte.driver import IGATransportDriver1D
from ttnte.xs import Material, Server
from ttnte.xs.benchmarks import pu239

DTYPE = torch.float64
DEVICE = torch.device("cpu")
OUTER_TOL = 1e-6


def _setup():
    mpi_context.init()
    torch.set_default_dtype(DTYPE)
    torch.autograd.set_grad_enabled(False)


def _assemble(driver, qset):
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-8
    config.cross.eps = config.rounding.eps
    config.max_dense_size = 0
    config.cross_jacobian_inverse = True
    driver.assemble(qset, config)


def _dd_solver(driver):
    config = DDSolverConfig(
        tol=1e-7,
        max_iter=40,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(config)
    strategy.set_local_solver(
        AMEnSolver(nswp=2, eps=1e-8, kickrank=2, local_iterations=60, resets=4)
    )
    return IGADDSolver(driver.mesh, strategy)


def _pu_slab(elements=5):
    """Two-patch homogeneous Pu-239 slab, vacuum ends (k ~ 1)."""
    _setup()
    fills, xs_server = pu239(num_groups=1, device=DEVICE, dtype=DTYPE)
    rc = 2.256751
    patches = [
        Patch.from_igakit(
            refine(line(a, b), elements, 3), device=DEVICE, dtype=DTYPE, fill=fills[0]
        )
        for a, b in [((-rc, 0), (0, 0)), ((0, 0), (rc, 0))]
    ]
    mesh = IGAMesh(mpi_context)
    for p in patches:
        mesh.add_block(p)
    mesh.connect()
    mesh.finalize()
    qset = QuadratureSet1D.gauss_legendre(32)
    qset.to_(DEVICE, DTYPE)
    driver = IGATransportDriver1D(mesh, xs_server, mpi_context)
    _assemble(driver, qset)
    return driver, [b.gid for b in driver.mesh.blocks]


def _absorber_slab():
    """One-patch pure absorber with an incident beam: a fixed-source problem
    with a closed-form answer, so 'same answer' is a meaningful check."""
    _setup()
    server = Server()
    mat = Material("Absorber")
    mat.chi = torch.zeros(1, dtype=DTYPE)
    mat.total = torch.tensor([0.75], dtype=DTYPE)
    mat.nu_fission = torch.zeros(1, dtype=DTYPE)
    mat.fission = torch.zeros(1, dtype=DTYPE)
    mat.absorption = torch.tensor([0.75], dtype=DTYPE)
    mat.scatter_gtg = torch.zeros((1, 1, 1), dtype=DTYPE)
    mat.finalize()
    server.add_material(mat)
    server.finalize()
    patch = Patch.from_igakit(
        refine(line((0, 0), (3.0, 0)), 12, 3),
        device=DEVICE,
        dtype=DTYPE,
        fill=mat.label,
    )
    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()
    patch.set_boundary_source(
        0, False, FixedSource(isotropic_strength=torch.tensor([2.0], dtype=DTYPE))
    )
    patch.set_boundary_type(0, True, BoundaryType.VACUUM)
    qset = QuadratureSet1D.gauss_legendre(32)
    qset.to_(DEVICE, DTYPE)
    driver = IGATransportDriver1D(mesh, server, mpi_context)
    _assemble(driver, qset)
    return driver, [b.gid for b in driver.mesh.blocks]


def _fields(result, gids):
    return {g: result.get_local_field(g) for g in gids}


def _rel_diff(a, b):
    a, b = a.to_dense(), b.to_dense()
    return float((a - b).norm() / b.norm())


# --------------------------------------------------------------- eigenvalue
def test_eigenvalue_warm_start_reconverges_immediately():
    driver, gids = _pu_slab()
    cold = driver.solve_eigenvalue(_dd_solver(driver), tol=OUTER_TOL, verbose=False)
    fields = _fields(cold, gids)

    driver2, _ = _pu_slab()
    warm = driver2.solve_eigenvalue(
        _dd_solver(driver2),
        tol=OUTER_TOL,
        verbose=False,
        initial_guess=fields,
        initial_guess_error=OUTER_TOL,
    )

    assert cold.num_outer_iterations > 4, "cold start should need real work"
    assert warm.num_outer_iterations <= 3
    assert warm.num_outer_iterations < cold.num_outer_iterations
    assert abs(warm.k_eff - cold.k_eff) < 5e-5
    for g in gids:
        assert _rel_diff(warm.get_local_field(g), cold.get_local_field(g)) < 1e-3
    # amplitude is consistent with the source normalization: the very first
    # outer iteration barely moves the scalar flux (a mis-scaled guess would
    # show O(1) here)
    assert warm.outer_flux_error[0] < 1e-3


def test_partial_guess_falls_back_to_ones_for_omitted_patches():
    """Only patch 0 gets a guess; patch 1 starts from ones.

    The solve must
    still converge to the same k. No `initial_guess_error` here: claiming the
    whole problem is accurate to 1e-6 would be false for patch 1 and just force
    tight truncation on a cold patch.
    """
    driver, gids = _pu_slab()
    cold = driver.solve_eigenvalue(_dd_solver(driver), tol=OUTER_TOL, verbose=False)

    driver2, _ = _pu_slab()
    part = driver2.solve_eigenvalue(
        _dd_solver(driver2),
        tol=OUTER_TOL,
        verbose=False,
        initial_guess={gids[0]: cold.get_local_field(gids[0])},
    )
    assert abs(part.k_eff - cold.k_eff) < 5e-5


# --------------------------------------------------------------- fixed source
def _absorber_error_vs_exact(driver, result):
    """Relative max error of the scalar flux against the closed form phi(x) = sum_{mu>0}
    w Q exp(-sigma_t x / mu) (sample points away from the incident face, where the
    boundary layer converges slowest)."""
    qset = QuadratureSet1D.gauss_legendre(32)
    qset.to_(DEVICE, DTYPE)
    mu, w = qset.points.reshape(-1), qset.weights.reshape(-1)
    inc = mu > 0
    patch = driver.mesh.blocks[0]
    field = result.compute_scalar_flux().get_local_field(patch.gid).to_dense()
    t = torch.linspace(0.0, 1.0, 9, dtype=DTYPE).reshape(-1, 1)
    values = patch.evaluate_field(field.reshape(-1, 1), t)[..., 0]
    x = patch.evaluate(t)[..., 0]
    exact = torch.stack(
        [(w[inc] * 2.0 * torch.exp(-0.75 * xi / mu[inc])).sum() for xi in x]
    )
    return float((values[1:] - exact[1:]).abs().max() / exact[1:].abs().max())


def test_fixed_source_warm_start_reconverges_immediately():
    """Judged against the exact solution, not against the cold run: two independently
    converged TT solves agree in the *scalar flux* to their truncation-limited accuracy,
    but their raw angular fluxes differ at the ~1e-2 level in components the scalar flux
    barely sees."""
    driver, gids = _absorber_slab()
    cold = driver.solve_fixed_source(_dd_solver(driver), tol=OUTER_TOL, verbose=False)
    fields = _fields(cold, gids)

    driver2, _ = _absorber_slab()
    warm = driver2.solve_fixed_source(
        _dd_solver(driver2),
        tol=OUTER_TOL,
        verbose=False,
        initial_guess=fields,
        initial_guess_error=OUTER_TOL,
    )

    assert warm.num_outer_iterations <= 3
    assert warm.num_outer_iterations < cold.num_outer_iterations
    err_cold = _absorber_error_vs_exact(driver, cold)
    err_warm = _absorber_error_vs_exact(driver2, warm)
    assert err_warm < 5e-3
    assert err_warm <= 1.5 * err_cold  # warm start must not degrade the answer


def test_good_guess_survives_default_forcing_once_interfaces_are_seeded():
    """The first sweep after a guess takes its interface data from the pending
    recv_buffers, which init_solver seeds from the guess.

    (Before seeding, the first sweeps saw no boundary data and destroyed a
    converged guess unless initial_guess_error also tightened the forcing; both
    variants now keep it.)
    """
    driver, gids = _pu_slab()
    cold = driver.solve_eigenvalue(_dd_solver(driver), tol=OUTER_TOL, verbose=False)
    fields = _fields(cold, gids)

    d_default, _ = _pu_slab()
    unseeded = d_default.solve_eigenvalue(
        _dd_solver(d_default), tol=OUTER_TOL, verbose=False, initial_guess=fields
    )
    d_seeded, _ = _pu_slab()
    seeded = d_seeded.solve_eigenvalue(
        _dd_solver(d_seeded),
        tol=OUTER_TOL,
        verbose=False,
        initial_guess=fields,
        initial_guess_error=OUTER_TOL,
    )

    for warm in (seeded, unseeded):
        assert warm.outer_flux_error[0] < 1e-3
        assert warm.num_outer_iterations <= 3
        assert warm.num_outer_iterations < cold.num_outer_iterations
        assert abs(warm.k_eff - cold.k_eff) < 5e-5


# --------------------------------------------------------------- validation
def test_unknown_gid_is_rejected():
    driver, gids = _pu_slab()
    cold = driver.solve_eigenvalue(_dd_solver(driver), tol=1e-4, verbose=False)
    driver2, _ = _pu_slab()
    bad = {max(gids) + 100: cold.get_local_field(gids[0])}
    with pytest.raises(RuntimeError, match="not a local patch"):
        driver2.solve_eigenvalue(
            _dd_solver(driver2), tol=OUTER_TOL, verbose=False, initial_guess=bad
        )


def test_mismatched_discretization_is_rejected():
    coarse_driver, gids = _pu_slab(elements=3)
    coarse = coarse_driver.solve_eigenvalue(
        _dd_solver(coarse_driver), tol=1e-4, verbose=False
    )
    fine_driver, _ = _pu_slab(elements=5)
    with pytest.raises(RuntimeError, match="mode sizes"):
        fine_driver.solve_eigenvalue(
            _dd_solver(fine_driver),
            tol=OUTER_TOL,
            verbose=False,
            initial_guess=_fields(coarse, gids),
        )


def test_no_guess_is_unchanged_default():
    """Omitting initial_guess (or passing an empty one) is the old behaviour."""
    a, gids = _pu_slab()
    ra = a.solve_eigenvalue(_dd_solver(a), tol=OUTER_TOL, verbose=False)
    b, _ = _pu_slab()
    rb = b.solve_eigenvalue(
        _dd_solver(b), tol=OUTER_TOL, verbose=False, initial_guess={}
    )
    assert ra.num_outer_iterations == rb.num_outer_iterations
    assert abs(ra.k_eff - rb.k_eff) < 1e-10


# ------------------------------------------------------ interface seeding
def test_initial_guess_seeds_interface_buffers():
    """A sweep takes interface data only from the pending recv_buffers, so a supplied
    guess must also fill them (with the face each neighbor would have sent); without a
    guess the buffers stay empty, as before."""
    driver, gids = _pu_slab()
    cold = driver.solve_eigenvalue(_dd_solver(driver), tol=OUTER_TOL, verbose=False)
    fields = _fields(cold, gids)

    d_none, _ = _pu_slab()
    dd_none = _dd_solver(d_none)
    d_none.init_solver(dd_none, clear_assemblers=False)
    for sys in dd_none.local_systems:
        assert all(not c.recv_buffer.defined() for c in sys.couplings)

    d_seed, _ = _pu_slab()
    dd_seed = _dd_solver(d_seed)
    d_seed.init_solver(dd_seed, clear_assemblers=False, initial_guess=fields)
    by_gid = {s.gid: s for s in dd_seed.local_systems}
    checked = 0
    for s in dd_seed.local_systems:
        for c in s.couplings:
            assert c.recv_buffer.defined()
            n = by_gid[c.connection.gid]
            sc = next(
                x
                for x in n.couplings
                if x.connection.gid == s.gid and x.fid == c.connection.fid
            )
            bdim = n.state.ndimension - len(sc.connection.mapping.flip) - 2 + sc.dim
            expected = n.state.narrow(bdim, -1 if sc.is_upper else 0, 1)
            assert _rel_diff(c.recv_buffer, expected) < 1e-6
            checked += 1
    assert checked >= 2
