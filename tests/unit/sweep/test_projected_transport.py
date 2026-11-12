"""ProjectedTransport: the bordered Galerkin projections of one group set's transport
operator onto a basis of group-set fluxes. Against the transport residual of the sweep
kernel (A_d x = r_d(0) - r_d(x), b_d = r_d(0)), the reduced system must be

    X^T A_d X  and  X^T b_d

to rounding for every direction, on flat and curved, conforming and non-conforming,
vacuum, reflective and cycle-broken meshes; bordering chunk by chunk (including the
lazily bordered curved-face caches) must agree with projecting at once, and the
flat-face path with quadrature."""

import pytest
import torch

from ttnte.sweep import SweepConfig

from . import test_nonconforming as nc
from .problems import BUILDERS, boxes, driver, quadrature, quarter_circle

G = 3
RANK = 5


def _setup(mesh, xs, num_dim, n=2, **kw):
    """Mid-solve state: nonzero, unconverged sources and lagged traces."""
    sw = driver(num_dim, mesh, xs)
    sw.setup(
        quadrature(num_dim, n), SweepConfig(tol=1e-14, max_iter=3, verbose=False, **kw)
    )
    sw.solve_fixed_source()
    return sw


def _basis(sw, groups, rank=RANK, seed=0):
    """A basis mixing swept fibers (realistic) and random vectors (every dof)."""
    nd = sw.ordinates.shape[0]
    dirs = list(range(0, nd, max(1, nd // 3)))[:2]
    fibers = sw.sweep_fibers([(dirs, groups)]).reshape(len(dirs), len(groups), -1)
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(rank - len(dirs), len(groups), sw.fiber_length, generator=gen)
    return torch.cat([fibers, noise])


def _reference(sw, d, groups, X):
    """X^T A_d X and X^T b_d from the transport residual."""
    flat = X.reshape(X.shape[0], -1)
    r0 = sw.transport_residual(d, groups, torch.zeros_like(X[0])).reshape(-1)
    cols = [r0 - sw.transport_residual(d, groups, x).reshape(-1) for x in X]
    return flat @ torch.stack(cols, 1), flat @ r0


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _check_exact(sw, groups, group_set=0, **kw):
    X = _basis(sw, groups)
    nd = sw.ordinates.shape[0]
    A, b, normals, curved = sw.projected_system(list(range(nd)), [X], group_set, **kw)
    worst = 0.0
    for d in range(nd):
        A_ref, b_ref = _reference(sw, d, groups, X)
        worst = max(worst, _rel(A[d], A_ref), _rel(b[d], b_ref))
    assert worst < 1e-12
    return normals, curved


CASES = [(d, r) for d in (1, 2, 3) for r in (False, True)]


@pytest.mark.parametrize("num_dim, reflective", CASES)
def test_reduced_system_is_exact(num_dim, reflective):
    mesh, xs = BUILDERS[num_dim](G, reflective)
    sw = _setup(mesh, xs, num_dim)
    _check_exact(sw, list(range(G)))


@pytest.mark.parametrize("reflective", [False, True])
def test_cycle_broken_mesh(reflective):
    """Lagged internal edges: their inflow is in b, not in A."""
    mesh, xs = quarter_circle(G, reflective, split=True)
    sw = _setup(mesh, xs, 2, n=4)
    assert sw.num_delayed_blocks > 0
    _check_exact(sw, list(range(G)))


@pytest.mark.parametrize("num_dim", [2, 3])
@pytest.mark.parametrize("reflective", [False, True])
def test_cartesian_flat_faces(num_dim, reflective):
    """Axis-aligned boxes: every face flat, one matrix per normal (2 ndim of them)."""
    mesh, xs = boxes(num_dim, G, reflective)
    sw = _setup(mesh, xs, num_dim)
    normals, curved = _check_exact(sw, list(range(G)))
    assert (normals, curved) == (2 * num_dim, 0)


@pytest.mark.parametrize(
    "surfaces, num_dim",
    [
        (lambda: nc.squares(False), 2),
        (lambda: nc.rings(False), 2),
        (lambda: nc.rings(False, flip=True), 2),
        (lambda: nc.cylinder_rings(False), 3),
    ],
    ids=["squares", "rings", "rings-flipped", "cylinder-rings"],
)
def test_nonconforming_interfaces(surfaces, num_dim):
    """The upwind side of a non-conforming face is the neighbor's own basis at the
    shared quadrature points (FaceLink coupling basis)."""
    fills, xs = nc.absorber()
    sw = _setup(nc.build(surfaces(), fills), xs, num_dim)
    _check_exact(sw, list(range(nc.G)))


def test_group_sets():
    """Each group set projects its own groups: sources, cross sections and traces."""
    mesh, xs = BUILDERS[2](G, True)
    sw = _setup(mesh, xs, 2, group_set_size=2)
    _check_exact(sw, [0, 1], group_set=0)
    _check_exact(sw, [2], group_set=1)


@pytest.mark.parametrize("num_dim", [2, 3])
def test_bordering_matches_projection_at_once(num_dim):
    """Chunk by chunk, assembling every direction in between (so the curved-face caches
    are bordered lazily too), equals one projection of the whole basis."""
    mesh, xs = BUILDERS[num_dim](G, True)
    sw = _setup(mesh, xs, num_dim)
    groups = list(range(G))
    X = _basis(sw, groups, rank=6)
    dirs = list(range(sw.ordinates.shape[0]))
    A1, b1, _, curved = sw.projected_system(dirs, [X])
    A2, b2, _, _ = sw.projected_system(
        dirs, [X[:1], X[1:3], X[3:4], X[4:]], assemble_each=True
    )
    assert curved > 0
    assert _rel(A2, A1) < 1e-13 and _rel(b2, b1) < 1e-13


@pytest.mark.parametrize("flat_faces", [True, False])
def test_reset_leaves_nothing_behind(flat_faces):
    """Reset() only rewinds the logical sizes; the storage keeps the old values.

    A smaller basis projected after it (so the stale entries sit inside its
    capacity, curved caches included) must equal a fresh projection, exactly.
    """
    mesh, xs = BUILDERS[2](G, True)
    sw = _setup(mesh, xs, 2)
    groups = list(range(G))
    old = _basis(sw, groups, rank=7, seed=3)
    X = _basis(sw, groups, rank=4, seed=4)
    dirs = list(range(sw.ordinates.shape[0]))
    kw = dict(flat_faces=flat_faces)
    A1, b1, _, curved = sw.projected_system(dirs, [X[:1], X[1:]], **kw)
    A2, b2, _, _ = sw.projected_system(dirs, [X[:1], X[1:]], previous=[old], **kw)
    assert curved > 0 and A1.shape == (len(dirs), 4, 4)
    assert torch.equal(A1, A2) and torch.equal(b1, b2)


@pytest.mark.parametrize(
    "build, num_dim",
    [
        (lambda: boxes(2, G, True), 2),
        (lambda: boxes(3, G, False), 3),
        (lambda: BUILDERS[1](G, True), 1),
        (lambda: BUILDERS[2](G, False), 2),
    ],
    ids=["boxes-2d", "boxes-3d", "slab", "quarter-circle"],
)
def test_flat_path_matches_quadrature(build, num_dim):
    """The flat-face fast path (one matrix per normal) equals projecting every face by
    quadrature per direction."""
    mesh, xs = build()
    sw = _setup(mesh, xs, num_dim)
    groups = list(range(G))
    X = _basis(sw, groups)
    dirs = list(range(sw.ordinates.shape[0]))
    A1, b1, normals, _ = sw.projected_system(dirs, [X])
    A2, b2, none, curved = sw.projected_system(dirs, [X], flat_faces=False)
    assert normals > 0 and none == 0 and curved > 0
    assert _rel(A1, A2) < 1e-13 and _rel(b1, b2) < 1e-13


def test_spanned_flux_is_reproduced():
    """With the exact fiber of direction d in the basis, the Galerkin solution of d's
    reduced system reproduces it (Galerkin projection of an exact solution)."""
    mesh, xs = BUILDERS[2](G, True)
    sw = _setup(mesh, xs, 2)
    groups = list(range(G))
    d = 3
    fiber = sw.sweep_fibers([([d], groups)]).reshape(1, G, -1)
    X = torch.cat([fiber, _basis(sw, groups, rank=4, seed=2)])
    A, b, _, _ = sw.projected_system([d], [X])
    c = torch.linalg.solve(A[0], b[0])
    psi = torch.einsum("j,jgn->gn", c, X)
    assert _rel(psi, fiber[0]) < 1e-10


def _final_fibers(sw):
    """(D, G, fiber_length) the angular flux of the last committed sweep
    (store_angular_flux), whose outgoing traces fill the lagged and reflective
    buffers."""
    nd, G_all = sw.ordinates.shape[0], G
    out = torch.zeros(nd, G_all, sw.fiber_length)
    for k, gid in enumerate(sw.local_gids):
        psi = sw.get_angular_flux(gid)
        n = psi.numel() // (nd * G_all)
        off = sw.fiber_offsets[k]
        out[:, :, off : off + n] = psi.reshape(nd, n, G_all).transpose(1, 2)
    return out


@pytest.mark.parametrize(
    "build, num_dim, set_size",
    [
        (lambda r: BUILDERS[1](G, r), 1, 0),
        (lambda r: boxes(2, G, r), 2, 0),
        (lambda r: boxes(3, G, r), 3, 0),
        (lambda r: boxes(2, G, r), 2, 2),
    ],
    ids=["slab", "boxes-2d", "boxes-3d", "boxes-2d-sets"],
)
def test_reflected_inflow_matches_dense_operator(build, num_dim, set_size):
    """The reflected part of X^T b_d, against a reference sharing none of the kernel's
    inflow code: the dense verification operator, whose reflective blocks (reflective
    minus vacuum mesh) are assembled separately, applied to the flux whose traces fill
    the buffers.

    Meshes without lagged edges, so the buffers hold reflected traces only.
    """
    sw = _setup(*build(True), num_dim, store_angular_flux=True, group_set_size=set_size)
    assert sw.graph.num_weight_lagged + sw.graph.num_cycle_lagged == 0
    vacuum = driver(num_dim, *build(False))
    vacuum.setup(quadrature(num_dim), SweepConfig(verbose=False))
    nd, G_all = sw.ordinates.shape[0], G
    psi = _final_fibers(sw)
    # Reflected inflow of every direction, (D, G, fiber_length)
    inflow = torch.zeros_like(psi)
    for k, gid in enumerate(sw.local_gids):
        R = sw.assemble_dense_operator(gid) - vacuum.assemble_dense_operator(gid)
        n = round((R.shape[0] // (nd * G_all)))
        R = R.reshape(nd, n, G_all, nd, n, G_all)
        off = sw.fiber_offsets[k]
        u = psi[:, :, off : off + n].transpose(1, 2)  # (D, n, G)
        inflow[:, :, off : off + n] = -torch.einsum("angbmh,bmh->agn", R, u)
    sets = [list(range(G))] if set_size == 0 else [[0, 1], [2]]
    for s, groups in enumerate(sets):
        X = torch.randn(
            4, len(groups), sw.fiber_length, generator=torch.Generator().manual_seed(s)
        )
        b = sw.buffered_projection(list(range(nd)), X, s)
        ref = torch.einsum("rgn,dgn->dr", X, inflow[:, groups])
        assert ref.norm() > 0
        assert _rel(b, ref) < 1e-12


@pytest.mark.parametrize(
    "split, reflective, n",
    [(False, False, 2), (False, True, 2), (True, False, 4), (True, True, 4)],
    ids=["weight-lagged", "weight-lagged-reflective", "cycles", "cycles-reflective"],
)
def test_lagged_inflow_matches_projected_traces(split, reflective, n):
    """The lagged part of X^T b_d, against the projection's own trace path. With the
    lagged edges made implicit, their inflow moves from b (kernel, read from the
    buffers) to A (at_points of the neighbor's basis), so with the flux that filled.

    the buffers as basis vector 0,
        b_default - b_implicit = (A_default - A_implicit) e_0.
    """
    mesh, xs = quarter_circle(G, reflective, split=split)
    sw = _setup(mesh, xs, 2, n=n, store_angular_flux=True)
    assert sw.graph.num_weight_lagged + sw.graph.num_cycle_lagged > 0
    groups = list(range(G))
    psi = _final_fibers(sw)
    lhs, rhs = [], []
    for d in range(sw.ordinates.shape[0]):
        X = torch.cat([psi[d : d + 1], _basis(sw, groups, rank=3, seed=d)])
        A0, b0, _, _ = sw.projected_system([d], [X])
        A1, b1, _, _ = sw.projected_system([d], [X], implicit_lagged=True)
        lhs.append(b0[0] - b1[0])
        rhs.append((A0[0] - A1[0])[:, 0])
    lhs, rhs = torch.stack(lhs), torch.stack(rhs)
    assert rhs.norm() > 0
    assert _rel(lhs, rhs) < 1e-12


def test_rejects_bad_direction():
    mesh, xs = BUILDERS[2](G, True)
    sw = _setup(mesh, xs, 2)
    nd = sw.ordinates.shape[0]
    X = _basis(sw, list(range(G)), rank=3)
    for d in (nd, -1):
        with pytest.raises(RuntimeError, match="Direction out of range"):
            sw.projected_system([d], [X])
        with pytest.raises(RuntimeError, match="Direction out of range"):
            sw.buffered_projection([d], X)


def test_rejects_bad_basis():
    mesh, xs = BUILDERS[1](G, False)
    sw = _setup(mesh, xs, 1)
    with pytest.raises(RuntimeError, match="fiber_length"):
        sw.projected_system([0], [torch.zeros(2, G, sw.fiber_length + 1)])
    with pytest.raises(RuntimeError, match="Group set out of range"):
        sw.projected_system([0], [torch.zeros(2, G, sw.fiber_length)], group_set=1)
