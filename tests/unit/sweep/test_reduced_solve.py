"""ProjectedTransport as a galerkin::ReducedOperator: the reduced solves (an LU per
direction, in parallel over directions), the residuals b_d - A_d X c_d through the sweep
kernel, and their error-weighted dual norms (FiberStore: R_P^{-T} in space, times the
angle and energy weights). Checked
against dense solves of the projected systems, Galerkin orthogonality, the step-3
residual of the reconstructed flux, single- against multi-threaded runs, and dense
mass-matrix references of every weighting factor."""

import itertools
import math

import pytest
import torch

from ttnte.sampling import Weighting
from ttnte.sweep import LocalSolverType, SweepConfig

from .problems import BUILDERS, boxes, driver, quadrature, quarter_circle

G = 3


def _setup(mesh, xs, num_dim, n=2, **kw):
    """Mid-solve state: nonzero, unconverged sources and lagged traces."""
    sw = driver(num_dim, mesh, xs)
    sw.setup(
        quadrature(num_dim, n), SweepConfig(tol=1e-14, max_iter=3, verbose=False, **kw)
    )
    sw.solve_fixed_source()
    return sw


def _basis(sw, groups, rank=5, seed=0):
    """Two swept fibers and random vectors (every dof exercised)."""
    nd = sw.ordinates.shape[0]
    dirs = list(range(0, nd, max(1, nd // 3)))[:2]
    fibers = sw.sweep_fibers([(dirs, groups)]).reshape(len(dirs), len(groups), -1)
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(rank - len(dirs), len(groups), sw.fiber_length, generator=gen)
    return torch.cat([fibers, noise])


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


CASES = {
    "slab-reflective": lambda: (_setup(*BUILDERS[1](G, True), 1), 1),
    "quarter-circle": lambda: (_setup(*BUILDERS[2](G, False), 2), 2),
    "octant-reflective": lambda: (_setup(*BUILDERS[3](G, True), 3), 3),
    "cycles-reflective": lambda: (
        _setup(*quarter_circle(G, True, split=True), 2, n=4),
        2,
    ),
    "boxes-3d": lambda: (_setup(*boxes(3, G, True), 3), 3),
}


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request):
    sw, _ = CASES[request.param]()
    groups = list(range(G))
    X = _basis(sw, groups)
    dirs = list(range(sw.ordinates.shape[0]))
    return sw, groups, X, dirs


def test_solve_matches_dense_solve(case):
    """C_d = (X^T A_d X)^{-1} X^T b_d, against torch's solve of the projected
    systems."""
    sw, _, X, dirs = case
    A, b, _, _ = sw.projected_system(dirs, [X])
    C = sw.galerkin_solve(dirs, [X])
    ref = torch.linalg.solve(A, b)
    assert C.shape == (len(dirs), X.shape[0])
    assert _rel(C, ref) < 1e-12


def test_galerkin_orthogonality(case):
    """The residual of the reduced solution is orthogonal to the basis: X^T r_d = X^T b_d
    - X^T A_d X c_d = 0. Ties solve() and residuals() to each other through two different
    assemblies (projection and sweep kernel)."""
    sw, _, X, dirs = case
    _, b, _, _ = sw.projected_system(dirs, [X])
    C = sw.galerkin_solve(dirs, [X])
    R = sw.galerkin_residuals(dirs, [X], C)
    XtR = torch.einsum("jgn,dgn->dj", X, R)
    assert (XtR.norm() / b.norm()).item() < 1e-11
    assert R.norm() > 1e-6 * b.norm()  # nonzero: the basis does not span psi


def test_residuals_match_transport_residual(case):
    """Residuals() is the step-3 residual of the reconstructed flux X^T c_d."""
    sw, groups, X, dirs = case
    gen = torch.Generator().manual_seed(7)
    C = torch.randn(len(dirs), X.shape[0], generator=gen)
    R = sw.galerkin_residuals(dirs, [X], C)
    for i, d in enumerate(dirs):
        psi = torch.einsum("j,jgn->gn", C[i], X)
        ref = sw.transport_residual(d, groups, psi)
        assert _rel(R[i], ref) < 1e-13


def test_parallel_matches_serial(case):
    """Directions solved and residuals evaluated concurrently (curved caches bordered by
    their own threads, the kernel's thread-local workspaces) equal a serial run,
    bitwise."""
    sw, _, X, dirs = case
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        C1 = sw.galerkin_solve(dirs, [X[:2], X[2:]])
        R1 = sw.galerkin_residuals(dirs, [X], C1)
        torch.set_num_threads(4)
        C4 = sw.galerkin_solve(dirs, [X[:2], X[2:]])
        R4 = sw.galerkin_residuals(dirs, [X], C4)
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(C1, C4) and torch.equal(R1, R4)


def test_spanned_flux_has_no_residual():
    """With the exact fiber of d in the basis, the reduced solution is that fiber and
    its residual is at rounding level."""
    mesh, xs = BUILDERS[2](G, True)
    sw = _setup(mesh, xs, 2)
    groups = list(range(G))
    d = 3  # not among _basis's swept fibers (0 and 5): those would repeat it
    fiber = sw.sweep_fibers([([d], groups)]).reshape(1, G, -1)
    X = torch.cat([fiber, _basis(sw, groups, rank=4, seed=3)])
    C = sw.galerkin_solve([d], [X])
    R = sw.galerkin_residuals([d], [X], C)
    r0 = sw.transport_residual(d, groups, torch.zeros_like(fiber[0]))
    assert (R.norm() / r0.norm()).item() < 1e-10


def test_singular_and_bad_arguments():
    mesh, xs = BUILDERS[2](G, False)
    sw = _setup(mesh, xs, 2)
    X = _basis(sw, list(range(G)), rank=3)
    nd = sw.ordinates.shape[0]
    with pytest.raises(RuntimeError, match="direction 1 is singular"):
        sw.galerkin_solve([1], [torch.cat([X, X[:1]])])
    with pytest.raises(RuntimeError, match="distinct"):
        sw.galerkin_solve([0, 0], [X])
    with pytest.raises(RuntimeError, match="Direction out of range"):
        sw.galerkin_solve([nd], [X])
    with pytest.raises(RuntimeError, match="Direction out of range"):
        sw.galerkin_residuals([-1], [X], torch.zeros(1, 3))
    with pytest.raises(RuntimeError, match="C must be"):
        sw.galerkin_residuals([0], [X], torch.zeros(1, 2))


# =================================================================
# Residual norms (FiberStore)


BOUNDS = [2.0e7, 1.0e5, 1.0, 1.0e-3]  # G + 1 descending: unequal lethargy widths


def _bounded(set_size=0):
    """The reflective quarter circle with energy group boundaries (lethargy widths)."""
    mesh, xs = BUILDERS[2](G, True)
    xs.group_bounds = BOUNDS
    return _setup(mesh, xs, 2, group_set_size=set_size)


def _reference_norms(sw, dirs, R, groups, weighting):
    """sqrt(sum_g (a_d e_g)^2 sum_u r^T M_u^{-1} r), from each unit's dense mass
    matrix."""
    bounds = torch.tensor(BOUNDS, dtype=torch.float64)
    du = torch.log(bounds[:-1] / bounds[1:])[groups]
    out = []
    for i, d in enumerate(dirs):
        total = 0.0
        for g in range(len(groups)):
            for k, gid in enumerate(sw.local_gids):
                disc = sw.get_discretization(gid)
                n = disc.num_dofs
                off = sw.fiber_offsets[k]
                r = R[i, g, off : off + n]
                if weighting.space:
                    M = disc.apply_mass(torch.eye(n, dtype=torch.float64))
                    total_u = (r @ torch.linalg.solve(M, r)).item()
                else:
                    total_u = (r @ r).item()
                scale = 1.0
                if weighting.angle:
                    scale *= sw.weights[d].item()
                if weighting.energy:
                    scale /= du[g].item()
                total += scale * total_u
        out.append(math.sqrt(total))
    return torch.tensor(out, dtype=torch.float64)


WEIGHTINGS = [
    Weighting(angle=a, energy=e, space=s)
    for a, e, s in itertools.product([False, True], repeat=3)
]


@pytest.mark.parametrize(
    "weighting",
    WEIGHTINGS,
    ids=[f"a{int(w.angle)}e{int(w.energy)}s{int(w.space)}" for w in WEIGHTINGS],
)
def test_residual_norms_every_factor(weighting):
    """Each factor, alone and combined, against dense references: Euclidean (no factor,
    Guo & Peng's l2), sqrt(r^T M^{-1} r) per unit (space), 1/sqrt(du_g) (energy),
    sqrt(w_d) (angle)."""
    sw = _bounded()
    dirs = [0, 3, 7]
    R = torch.randn(
        len(dirs), G, sw.fiber_length, generator=torch.Generator().manual_seed(1)
    )
    norms, effective = sw.residual_norms(dirs, R, 0, weighting)
    assert (effective.angle, effective.energy, effective.space) == (
        weighting.angle,
        weighting.energy,
        weighting.space,
    )
    ref = _reference_norms(sw, dirs, R, list(range(G)), weighting)
    assert _rel(norms, ref) < 1e-12
    if not (weighting.angle or weighting.energy or weighting.space):
        assert _rel(norms, R.flatten(1).norm(dim=1)) < 1e-14


def test_residual_norms_group_sets():
    """Each set's store weights its own groups by their own lethargy widths."""
    sw = _bounded(set_size=2)
    dirs = [1, 2]
    full = Weighting()
    for s, groups in enumerate([[0, 1], [2]]):
        R = torch.randn(
            len(dirs),
            len(groups),
            sw.fiber_length,
            generator=torch.Generator().manual_seed(s),
        )
        norms, _ = sw.residual_norms(dirs, R, s, full)
        ref = _reference_norms(sw, dirs, R, groups, full)
        assert _rel(norms, ref) < 1e-12


def test_residual_norms_banded_path():
    """Patches above the dense-factor limit (TTNTE_DENSE_MASS_MAX, 3000 dofs) take the
    banded path (R^{-T} = L^{-1}, LAPACK dtbsv): a 1-D slab of two 3102-dof patches."""
    from igakit import cad

    from ttnte import mpi_context
    from ttnte.cad import Patch
    from ttnte.mesh import IGAMesh

    from .problems import CPU, _finish, _source, multigroup_xs

    fills, xs = multigroup_xs(G)
    mesh = IGAMesh(mpi_context)
    for k, (x0, x1) in enumerate([(0.0, 1.0), (1.0, 2.0)]):
        line = cad.refine(cad.line((x0, 0), (x1, 0)), 3100, 2)  # 3102 dofs
        patch = Patch.from_igakit(
            line, device=CPU, dtype=torch.float64, fill=fills[k].label
        )
        if k == 0:
            patch.source = _source(G)
        mesh.add_block(patch)
    xs.group_bounds = BOUNDS
    # BANDED_LU: SYLVESTER's setup on patches this large takes minutes
    sw = _setup(_finish(mesh, 1, True), xs, 1, local_solver=LocalSolverType.BANDED_LU)
    assert min(sw.get_discretization(g).num_dofs for g in sw.local_gids) > 3000
    dirs = [0, 1]
    R = torch.randn(
        len(dirs), G, sw.fiber_length, generator=torch.Generator().manual_seed(2)
    )
    norms, _ = sw.residual_norms(dirs, R, 0, Weighting())
    ref = _reference_norms(sw, dirs, R, list(range(G)), Weighting())
    assert _rel(norms, ref) < 1e-11


def test_residual_norms_reject_bad_input():
    sw = _bounded()
    R = torch.zeros(2, G, sw.fiber_length)
    with pytest.raises(RuntimeError, match="residuals must be"):
        sw.residual_norms([0], R)
    with pytest.raises(RuntimeError, match="Direction out of range"):
        sw.residual_norms([0, sw.ordinates.shape[0]], R)
