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
from .problems import (
    BUILDERS,
    boxes,
    driver,
    mid_solve,
    quadrature,
    quarter_circle,
    rel,
)

G = 3


def _anisotropic_xs(num_groups, fissile=False):
    """problems.multigroup_xs (non-fissile) with a P1 transfer, 0.3 times P0: two
    Legendre orders."""
    materials, _ = problems.multigroup_xs(num_groups)
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


def _build(builder, *args, anisotropic=False, **kw):
    if anisotropic:
        kw["xs_factory"] = _anisotropic_xs
    return builder(*args, **kw)


def _setup(mesh, xs, num_dim, n=2, **kw):
    sw = driver(num_dim, mesh, xs)
    sw.setup(quadrature(num_dim, n), SweepConfig(verbose=False, **kw))
    sw.solve_fixed_source()
    return sw


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


CASES = {
    "slab-reflective": lambda: (mid_solve(*BUILDERS[1](G, True), 1), 1),
    "quarter-circle": lambda: (mid_solve(*BUILDERS[2](G, False), 2), 2),
    "cycles-reflective": lambda: (
        mid_solve(*quarter_circle(G, True, split=True), 2, n=4),
        2,
    ),
    "anisotropic": lambda: (
        mid_solve(*_build(BUILDERS[2], G, True, anisotropic=True), 2),
        2,
    ),
    "boxes-3d": lambda: (mid_solve(*boxes(3, G, True), 3), 3),
    # 144 directions: more than one block of 64 (pass 1's factors are kept
    # per direction across blocks for pass 2)
    "many-directions": lambda: (mid_solve(*BUILDERS[2](G, False), 2, n=6), 2),
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
    assert rel(XSphi, ref) < 1e-12
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
    assert rel(L, ref) < 1e-11
    assert rel(Lam, wY.t() @ ref) < 1e-11


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
    assert rel(psi, fibers) < 1e-8
    phi = torch.einsum("dm,dgn->mgn", sw.harmonics * sw.weights[:, None], fibers)
    assert rel(torch.einsum("mj,jgn->mgn", Lam, X), phi) < 1e-8
    # the fixed source differs from the full one: there is scattering to remove
    _, src, XSphi, _ = sw.galerkin_scattering([X], group_set=s)
    assert XSphi.norm() > 1e-3 * src.norm()


def test_rejects_bad_shapes():
    sw = mid_solve(*BUILDERS[2](G, False), 2)
    X = _random_basis(sw, list(range(G)), rank=3)
    with pytest.raises(RuntimeError, match="F must be"):
        sw.galerkin_angular([X], F=torch.zeros(1, 2))
    with pytest.raises(RuntimeError, match="phi must be"):
        sw.galerkin_scattering([X], phi=torch.zeros(1, G, 2))


# ----------------------------------------------------------------------
# Pairs (Options::spatial_basis, ours: DEIM over (direction, group) units): a
# spatial basis, per order and pair of groups the block K_l(g, g') = X^T
# sigma_{s,l}(g <- g') M X, and the angular problem over every unit, the groups
# coupled through the blocks: (M G r) unknowns Lam_{m,g}.


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def pair_case(request):
    sw, _ = CASES[request.param]()
    return sw, _random_basis(sw, [0], rank=5)


def _place(X, ng):
    """Direction-sampling basis (r ng, ng, Ns): spatial vector j in group g' only,
    vector j ng + g'."""
    r, _, n = X.shape
    out = torch.zeros(r, ng, ng, n)
    for g in range(ng):
        out[:, g, g] = X[:, 0]
    return out.reshape(r * ng, ng, n)


def test_pair_blocks_match_direction_projection(pair_case):
    """K_l(g, g') equals the direction-sampling K_l on group-local vectors (block ((j,
    g), (i, g')) = x_j^T sigma(g <- g') M x_i, through the dense transfer, not the per-
    block one); bordered in chunks as at once; symmetric blocks; zero exactly where no
    unit transfers."""
    sw, X = pair_case
    r = X.shape[0]
    K, _, _, _ = sw.galerkin_scattering([X[:2], X[2:]], spatial_basis=True)
    K1, _, _, _ = sw.galerkin_scattering([X], spatial_basis=True)
    assert torch.allclose(K, K1, rtol=0, atol=1e-13 * K1.abs().max())
    Kd, _, _, _ = sw.galerkin_scattering([_place(X, G)])
    ref = Kd.reshape(-1, r, G, r, G).permute(0, 2, 4, 1, 3)  # (L, g, g', r, r)
    assert rel(K, ref) < 1e-12
    assert torch.allclose(K, K.transpose(3, 4), rtol=0, atol=1e-13 * K.abs().max())
    zero = K.flatten(3).norm(dim=3) == 0
    assert zero.any() and not zero.all()  # no upscatter beyond one group


def test_pair_projections_match_direction_projection(pair_case):
    """X^T s and X^T S phi per (moment, group) row equal the direction-sampling
    projections on group-local vectors."""
    sw, X = pair_case
    r = X.shape[0]
    M = sw.harmonics.shape[1]
    phi = torch.randn(M, G, sw.fiber_length, generator=torch.Generator().manual_seed(3))
    _, src, XSphi, _ = sw.galerkin_scattering([X], phi=phi, spatial_basis=True)
    _, src_d, XSphi_d, _ = sw.galerkin_scattering([_place(X, G)], phi=phi)
    # direction row m, column j G + g' -> pair row m G + g', column j
    to_pairs = lambda t: t.reshape(M, r, G).transpose(1, 2).reshape(M * G, r)
    assert rel(src, to_pairs(src_d)) < 1e-12
    assert rel(XSphi, to_pairs(XSphi_d)) < 1e-12


def test_pair_angular_matches_dense_solve(pair_case):
    """L solves A_u L_u - sum_m Y_m(d) sum_g' K_{l(m)}(g, g') Lam_{m,g'} = f_u,
    Lam_{m,g} = sum_d w_d Y_m(d) L_{d,g}, assembled densely from the units'
    projected systems."""
    sw, X = pair_case
    D, r = sw.ordinates.shape[0], X.shape[0]
    U = D * G
    A, b, _, _, _ = sw.projected_system(list(range(U)), [X], spatial_basis=True)
    K, src, _, _ = sw.galerkin_scattering([X], spatial_basis=True)
    Y = sw.harmonics
    wY = Y * sw.weights[:, None]
    M = Y.shape[1]
    orders = sw.moment_order.tolist()
    F = torch.randn(M * G, r, generator=torch.Generator().manual_seed(2))
    L, Lam, _ = sw.galerkin_angular([X[:2], X[2:]], F=F, spatial_basis=True)

    E = (F - src).reshape(M, G, r)
    big = torch.zeros(U * r, U * r)
    rhs = torch.zeros(U * r)
    for u in range(U):
        d, g = divmod(u, G)
        big[u * r : (u + 1) * r, u * r : (u + 1) * r] = A[u]
        rhs[u * r : (u + 1) * r] = b[u] + Y[d] @ E[:, g]
        for m, l in enumerate(orders):
            if l >= K.shape[0]:
                continue
            for dp in range(D):
                for gp in range(G):
                    up = dp * G + gp
                    big[u * r : (u + 1) * r, up * r : (up + 1) * r] -= (
                        Y[d, m] * wY[dp, m] * K[l, g, gp]
                    )
    ref = torch.linalg.solve(big, rhs).reshape(U, r)
    assert rel(L, ref) < 1e-11
    lam = torch.einsum("dm,dgr->mgr", wY, ref.reshape(D, G, r)).reshape(M * G, r)
    assert rel(Lam, lam) < 1e-11


def test_pair_angular_thread_independent(pair_case):
    sw, X = pair_case
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        L1, Lam1, _ = sw.galerkin_angular([X], spatial_basis=True)
        torch.set_num_threads(4)
        L4, Lam4, _ = sw.galerkin_angular([X], spatial_basis=True)
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(L1, L4) and torch.equal(Lam1, Lam4)


@pytest.mark.parametrize("name", list(EXACT))
def test_pair_exact_basis_reproduces_flux(name):
    """Converged source iteration, spatial basis = orthonormal span of every unit's
    exact fiber: the pair angular solve with the fixed source returns the fibers and
    each group's moments."""
    sw, s = EXACT[name]()
    groups = _groups(sw, s)
    ng = len(groups)
    D = sw.ordinates.shape[0]
    fibers = sw.sweep_fibers([(list(range(D)), groups)]).reshape(D * ng, -1)
    _, S, Vh = torch.linalg.svd(fibers, full_matrices=False)
    k = int((S > 1e-10 * S[0]).sum())
    X = Vh[:k].reshape(k, 1, -1)
    L, Lam, _ = sw.galerkin_angular([X], group_set=s, spatial_basis=True)
    assert rel(L @ X[:, 0], fibers) < 1e-8
    phi = torch.einsum(
        "dm,dgn->mgn", sw.harmonics * sw.weights[:, None], fibers.reshape(D, ng, -1)
    )
    M = phi.shape[0]
    assert rel((Lam @ X[:, 0]).reshape(M, ng, -1), phi) < 1e-8


@pytest.mark.parametrize("num_dim", [1, 2])
def test_one_group_pairs_equal_directions(num_dim):
    """With one group a pair is a direction: the same K, projections and angular
    solution."""
    sw = mid_solve(*BUILDERS[num_dim](1, True), num_dim)
    X = torch.randn(4, 1, sw.fiber_length, generator=torch.Generator().manual_seed(4))
    Kp, srcp, XSp, _ = sw.galerkin_scattering([X], spatial_basis=True)
    Kd, srcd, XSd, _ = sw.galerkin_scattering([X])
    assert rel(Kp[:, 0, 0], Kd) < 1e-13
    assert rel(srcp, srcd) < 1e-13 and rel(XSp, XSd) < 1e-13
    Lp, Lamp, _ = sw.galerkin_angular([X], spatial_basis=True)
    Ld, Lamd, _ = sw.galerkin_angular([X])
    assert rel(Lp, Ld) < 1e-12 and rel(Lamp, Lamd) < 1e-12


def test_pair_rejects_bad_shapes():
    sw = mid_solve(*BUILDERS[2](G, False), 2)
    X = _random_basis(sw, [0], rank=3)
    M = sw.harmonics.shape[1]
    with pytest.raises(RuntimeError, match="F must be"):
        sw.galerkin_angular([X], F=torch.zeros(M, 3), spatial_basis=True)
