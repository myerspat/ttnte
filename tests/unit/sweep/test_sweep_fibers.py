"""Fiber sweeps: any (direction, group) pairs swept exactly with frozen sources and
lagged traces, without touching the iterate."""

import pytest
import torch

from ttnte.sweep import LocalSolverType, SweepConfig

from .problems import BUILDERS, driver, quadrature

G = 3


def _solved(num_dim, reflective, threads=1, **kw):
    mesh, xs = BUILDERS[num_dim](G, reflective)
    sw = driver(num_dim, mesh, xs)
    cfg = SweepConfig(
        tol=1e-13,
        max_iter=5000,
        verbose=False,
        store_angular_flux=True,
        num_threads=threads,
        **kw,
    )
    sw.setup(quadrature(num_dim), cfg)
    sw.solve_fixed_source()
    return sw


def _psi(sw):
    """(N_Omega, G, fiber_length) angular flux of the last sweep."""
    nd = sw.ordinates.shape[0]
    blocks = [sw.get_angular_flux(gid) for gid in sw.local_gids]
    return torch.cat([b.reshape(nd, -1, G).permute(0, 2, 1) for b in blocks], 2)


CASES = [(d, r) for d in (1, 2, 3) for r in (False, True)]


@pytest.mark.parametrize("num_dim, reflective", CASES)
def test_all_fibers_match_angular_flux(num_dim, reflective):
    sw = _solved(num_dim, reflective)
    psi = _psi(sw)
    nd = psi.shape[0]
    fibers = sw.sweep_fibers([(list(range(nd)), list(range(G)))])
    fibers = fibers.reshape(nd, G, -1)
    err = ((fibers - psi).norm() / psi.norm()).item()
    # Lagged traces were updated by the last commit: agreement is at the
    # convergence tolerance
    assert err < 1e-10


@pytest.mark.parametrize("num_dim, reflective", CASES)
def test_blocks_are_independent(num_dim, reflective):
    """Any split of the pairs into blocks gives bitwise the same fibers."""
    sw = _solved(num_dim, reflective)
    nd = sw.ordinates.shape[0]
    whole = sw.sweep_fibers([(list(range(nd)), list(range(G)))]).reshape(nd, G, -1)
    rng = torch.Generator().manual_seed(0)
    dirs = torch.randperm(nd, generator=rng).tolist()
    half = max(1, nd // 2)
    blocks = [(dirs[:half], [2, 0]), (dirs[half:], [1]), (dirs[half:], [0, 2])]
    out = sw.sweep_fibers(blocks)
    row = 0
    for ds, gs in blocks:
        for d in ds:
            for g in gs:
                assert torch.equal(out[row], whole[d, g])
                row += 1
    rows = sw.sweep_fibers([(dirs[:half], [1])])
    for i, d in enumerate(dirs[:half]):
        assert torch.equal(rows[i], whole[d, 1])


@pytest.mark.parametrize("num_dim", [2, 3])
def test_threads_and_granularity_agree(num_dim):
    serial = _solved(num_dim, True)
    nd = serial.ordinates.shape[0]
    request = [(list(range(nd)), list(range(G)))]
    reference = serial.sweep_fibers(request)
    threaded = _solved(num_dim, True, threads=4, directions_per_task=1)
    assert torch.allclose(threaded.sweep_fibers(request), reference, rtol=0, atol=1e-11)


def test_iterate_is_unchanged():
    sw = _solved(2, True)
    before = [sw.get_moments(gid).clone() for gid in sw.local_gids]
    sw.sweep_fibers([([0, 1], [0, 1, 2])])
    for gid, m in zip(sw.local_gids, before):
        assert torch.equal(sw.get_moments(gid), m)
    assert sw.num_fibers == 6


def test_dense_solver_and_layout():
    sw = _solved(3, False, local_solver=LocalSolverType.DENSE_LU)
    assert sw.fiber_offsets[0] == 0
    assert sw.fiber_length == sum(
        sw.get_discretization(gid).num_dofs for gid in sw.local_gids
    )
    with pytest.raises(RuntimeError):
        sw.sweep_fibers([([10**6], [0])])
