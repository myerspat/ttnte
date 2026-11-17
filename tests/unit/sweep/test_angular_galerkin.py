"""ProjectedTransport with Options::scattering (Han et al. 2026, step H2): the projected
within-set scattering K_l = X^T S_l X (bordered with the volume terms), X^T S phi of a
flux, and the angular Galerkin problem (eq. 3.4 with angular moments) that couples every
direction through its own scattering.

Checked by consistency of the bordered K_l with the direct X^T S phi, by a dense solve
of the coupled reduced system, and by exactness: on a converged source iteration, with a
basis spanning every direction's exact fiber, the angular solve reproduces the fibers and
their moments (which needs K_l and X^T S phi to be the solver's own scattering)."""

import pytest
import torch

from ttnte.sweep import InnerSolver, SweepConfig
from ttnte.xs import Material, Server

from . import problems
from .problems import BUILDERS, boxes, driver, quadrature, quarter_circle

G = 3


def _anisotropic_xs(num_groups, fissile=False):
    """problems.multigroup_xs (non-fissile) with a P1 transfer, 0.3 times P0: two
    Legendre orders."""
    materials, _ = _isotropic_xs(num_groups)
    server = Server()
    out = []
    for name, old in zip(["Fuel", "Moderator"], materials):
        m = Material(name)
        m.total = old.total
        p0 = old.scatter_gtg
        m.scatter_gtg = torch.cat([p0, 0.3 * p0])
        m.absorption = old.total - p0[0].sum(1)  # finalize() would sum P1 too
        m.finalize()
        server.add_material(m)
        out.append(m)
    server.finalize()
    return out, server


_isotropic_xs = problems.multigroup_xs


def _build(builder, *args, anisotropic=False, **kw):
    if not anisotropic:
        return builder(*args, **kw)
    problems.multigroup_xs = _anisotropic_xs
    try:
        return builder(*args, **kw)
    finally:
        problems.multigroup_xs = _isotropic_xs


def _setup(mesh, xs, num_dim, n=2, **kw):
    sw = driver(num_dim, mesh, xs)
    sw.setup(quadrature(num_dim, n), SweepConfig(verbose=False, **kw))
    sw.solve_fixed_source()
    return sw


def _mid_solve(mesh, xs, num_dim, n=2, **kw):
    """Nonzero, unconverged sources and lagged traces."""
    return _setup(mesh, xs, num_dim, n, tol=1e-14, max_iter=3, **kw)


def _converged(mesh, xs, num_dim, n=2, **kw):
    """Source iteration to rounding: the sweep of the current source is the exact
    flux, and the solver's moments are the ones that made that source."""
    kw = dict(inner_solver=InnerSolver.RICHARDSON, tol=1e-13, max_iter=2000) | kw
    return _setup(mesh, xs, num_dim, n, **kw)


def _groups(sw, group_set):
    size = sw.config.group_set_size or G
    return list(range(group_set * size, min(G, (group_set + 1) * size)))


def _random_basis(sw, groups, rank=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rank, len(groups), sw.fiber_length, generator=gen)


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


CASES = {
    "slab-reflective": lambda: (_mid_solve(*BUILDERS[1](G, True), 1), 1),
    "quarter-circle": lambda: (_mid_solve(*BUILDERS[2](G, False), 2), 2),
    "cycles-reflective": lambda: (
        _mid_solve(*quarter_circle(G, True, split=True), 2, n=4),
        2,
    ),
    "anisotropic": lambda: (
        _mid_solve(*_build(BUILDERS[2], G, True, anisotropic=True), 2),
        2,
    ),
    "boxes-3d": lambda: (_mid_solve(*boxes(3, G, True), 3), 3),
    # 144 directions: more than one block of 64 (pass 1's factors are kept
    # per direction across blocks for pass 2)
    "many-directions": lambda: (_mid_solve(*BUILDERS[2](G, False), 2, n=6), 2),
}


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request):
    sw, _ = CASES[request.param]()
    return sw, _random_basis(sw, list(range(G)))


def test_bordered_scattering_matches_direct(case):
    """K_l bordered chunk by chunk in extend() equals X^T S phi applied directly to a
    flux in the span: phi_m = sum_j Lam_{m,j} x_j gives K_{l(m)} Lam_m."""
    sw, X = case
    K, _, _, _ = sw.galerkin_scattering([X[:2], X[2:]])
    K1, _, _, _ = sw.galerkin_scattering([X])
    assert torch.allclose(K, K1, rtol=0, atol=1e-13 * K1.abs().max())
    M = sw.harmonics.shape[1]
    Lam = torch.randn(M, X.shape[0], generator=torch.Generator().manual_seed(1))
    phi = torch.einsum("mj,jgn->mgn", Lam, X)
    _, _, XSphi, _ = sw.galerkin_scattering([X], phi=phi)
    ref = torch.stack(
        [
            K[l] @ Lam[m] if l < K.shape[0] else torch.zeros(X.shape[0])
            for m, l in enumerate(sw.moment_order.tolist())
        ]
    )
    assert _rel(XSphi, ref) < 1e-12
    assert K.norm() > 0


def test_scattering_is_nonsymmetric_with_upscatter(case):
    """Upscatter makes sigma_s(g <- g') != sigma_s(g' <- g): K_l is not symmetric
    (exercises the separate new rows)."""
    sw, X = case
    K, _, _, _ = sw.galerkin_scattering([X[:3], X[3:]])
    assert (K - K.transpose(1, 2)).norm() > 1e-6 * K.norm()


def test_angular_galerkin_matches_dense_solve(case):
    """L solves the coupled system A_d L_d - sum_m Y_m(d) K_{l(m)} Lam_m = f_d, Lam_m =
    sum_d w_d Y_m(d) L_d, assembled densely from the projected systems."""
    sw, X = case
    dirs = list(range(sw.ordinates.shape[0]))
    D, r = len(dirs), X.shape[0]
    A, b, _, _, _ = sw.projected_system(dirs, [X])
    K, src, _, _ = sw.galerkin_scattering([X])
    Y = sw.harmonics
    wY = Y * sw.weights[:, None]
    M = Y.shape[1]
    orders = sw.moment_order.tolist()
    F = torch.randn(M, r, generator=torch.Generator().manual_seed(2))
    L, Lam, _ = sw.galerkin_angular([X[:2], X[2:]], F=F)

    big = torch.zeros(D * r, D * r)
    rhs = torch.zeros(D * r)
    for d in range(D):
        big[d * r : (d + 1) * r, d * r : (d + 1) * r] = A[d]
        rhs[d * r : (d + 1) * r] = b[d] + Y[d] @ (F - src)
        for m, l in enumerate(orders):
            if l >= K.shape[0]:
                continue
            for dp in range(D):
                big[d * r : (d + 1) * r, dp * r : (dp + 1) * r] -= (
                    Y[d, m] * wY[dp, m] * K[l]
                )
    ref = torch.linalg.solve(big, rhs).reshape(D, r)
    assert _rel(L, ref) < 1e-11
    assert _rel(Lam, wY.t() @ ref) < 1e-11


def test_angular_galerkin_thread_independent(case):
    """Directions in fixed blocks: one thread and four give the same bits."""
    sw, X = case
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        L1, Lam1, _ = sw.galerkin_angular([X])
        torch.set_num_threads(4)
        L4, Lam4, _ = sw.galerkin_angular([X])
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(L1, L4) and torch.equal(Lam1, Lam4)


EXACT = {
    "slab-reflective": lambda: (_converged(*BUILDERS[1](G, True), 1), 0),
    "quarter-circle": lambda: (_converged(*BUILDERS[2](G, False), 2), 0),
    "cycles-reflective": lambda: (
        _converged(*quarter_circle(G, True, split=True), 2, n=4),
        0,
    ),
    "anisotropic-1d": lambda: (
        _converged(*_build(BUILDERS[1], G, True, anisotropic=True), 1),
        0,
    ),
    "anisotropic-2d": lambda: (
        _converged(*_build(BUILDERS[2], G, True, anisotropic=True), 2),
        0,
    ),
    "octant-reflective": lambda: (_converged(*BUILDERS[3](G, True), 3), 0),
    "two-sets-first": lambda: (
        _converged(*BUILDERS[2](G, True), 2, group_set_size=2),
        0,
    ),
    "two-sets-second": lambda: (
        _converged(*BUILDERS[2](G, True), 2, group_set_size=2),
        1,
    ),
}


@pytest.mark.parametrize("name", list(EXACT))
def test_exact_basis_reproduces_flux(name):
    """Converged source iteration, basis = orthonormal span of every direction's exact
    fiber: the angular solve with F = X^T s - X^T S phi (the fixed source: the
    solver's source without its within-set scattering) returns the fibers and their
    moments."""
    sw, s = EXACT[name]()
    groups = _groups(sw, s)
    D = sw.ordinates.shape[0]
    fibers = sw.sweep_fibers([(list(range(D)), groups)]).reshape(D, len(groups), -1)
    U, S, Vh = torch.linalg.svd(fibers.reshape(D, -1), full_matrices=False)
    k = int((S > 1e-10 * S[0]).sum())
    X = Vh[:k].reshape(k, len(groups), -1)
    L, Lam, F = sw.galerkin_angular([X], group_set=s)
    psi = torch.einsum("dj,jgn->dgn", L, X)
    assert _rel(psi, fibers) < 1e-8
    phi = torch.einsum("dm,dgn->mgn", sw.harmonics * sw.weights[:, None], fibers)
    assert _rel(torch.einsum("mj,jgn->mgn", Lam, X), phi) < 1e-8
    # the fixed source differs from the full one: there is scattering to remove
    _, src, XSphi, _ = sw.galerkin_scattering([X], group_set=s)
    assert XSphi.norm() > 1e-3 * src.norm()


def test_rejects_bad_shapes():
    sw = _mid_solve(*BUILDERS[2](G, False), 2)
    X = _random_basis(sw, list(range(G)), rank=3)
    with pytest.raises(RuntimeError, match="F must be"):
        sw.galerkin_angular([X], F=torch.zeros(1, 2))
    with pytest.raises(RuntimeError, match="phi must be"):
        sw.galerkin_scattering([X], phi=torch.zeros(1, G, 2))
