"""The iterative sweep solvers converge to the direct solution of the assembled
system: on a single curved patch with reflective faces (lagged in the sweep,
implicit in the dense operator) and 7-group cross sections with upscatter,
every combination of group iteration and inner solver must reproduce the dense
fixed-source solution and the dominant eigenvalue of A^{-1} F."""

import pytest
import torch
from igakit.cad import circle, coons, line, refine

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.math import ProductQuadrature
from ttnte.mesh import IGAMesh
from ttnte.physics import BCPlane, BoundaryType, FixedSource
from ttnte.sweep import GroupIteration, IGASweepDriver2D, InnerSolver, SweepConfig
from ttnte.xs.benchmarks import kaist

CPU = torch.device("cpu")
VARIANTS = [
    dict(
        group_iteration=GroupIteration.JACOBI,
        inner_solver=InnerSolver.RICHARDSON,
        inner_max_iter=1,
    ),
    dict(inner_solver=InnerSolver.RICHARDSON, group_set_size=1),
    dict(),
    dict(group_set_size=1),
    dict(group_set_size=3, group_iteration=GroupIteration.JACOBI),
    dict(inner_forcing=0.1),
    dict(directions_per_task=0),
    dict(directions_per_task=1, num_threads=4),
    dict(directions_per_task=3, num_threads=4, cache_factorizations=True),
]


def _mesh(fill, source=None):
    r = 1.0
    c0 = circle(r, angle=torch.pi / 4)
    c1 = circle(r, angle=-torch.pi / 4).rotate(torch.pi / 2)
    surface = coons([[line((0, 0), (0, r)), c0], [line((0, 0), (r, 0)), c1]])
    patch = Patch.from_igakit(
        refine(surface, 1, 2), device=CPU, dtype=torch.float64, fill=fill
    )
    if source is not None:
        patch.source = FixedSource(isotropic_strength=source)
    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True), BoundaryType.REFLECTIVE, tol=1e-9
    )
    mesh.finalize()
    return mesh


def _qset():
    qset = ProductQuadrature.gauss_legendre_chebyshev(2, 2, 2)
    qset.to_(CPU, torch.float64)
    return qset


def _driver(mesh, xs, **kw):
    sw = IGASweepDriver2D(mesh, xs, mpi_context)
    sw.setup(_qset(), SweepConfig(tol=1e-11, max_iter=20000, verbose=False, **kw))
    return sw


def _scalar_flux(x, sw):
    """(D, N, G) -> scalar flux (*shape, G)."""
    D = sw.weights.numel()
    x = x.reshape(D, -1, x.shape[-1])
    return torch.einsum("d,dng->ng", sw.weights, x)


@pytest.mark.parametrize("kw", VARIANTS)
def test_fixed_source_matches_direct_solve(kw):
    torch.set_default_dtype(torch.float64)
    fills, xs = kaist(problem="2B", device=CPU, dtype=torch.float64)
    G = xs.num_groups
    source = torch.linspace(1.0, 2.0, G)
    sw = _driver(_mesh(fills[8], source), xs, **kw)  # water: upscatter
    sw.solve_fixed_source()
    assert sw.converged

    ref = _driver(_mesh(fills[8], source), xs)
    A = ref.assemble_dense_operator(0)
    disc = ref.get_discretization(0)
    D, N = ref.weights.numel(), disc.num_dofs
    q = (source.view(G, 1) * disc.basis_integral.view(1, N)).T  # (N, G) load
    b = q.unsqueeze(0).expand(D, N, G).reshape(-1)
    x = torch.linalg.solve(A, b).reshape(D, N, G)
    phi = _scalar_flux(x, ref)
    got = sw.get_scalar_flux(0).reshape(N, G)
    assert ((got - phi).norm() / phi.norm()).item() < 1e-8


@pytest.mark.parametrize("kw", VARIANTS)
def test_eigenvalue_matches_dense_eigensolve(kw):
    torch.set_default_dtype(torch.float64)
    fills, xs = kaist(problem="2B", device=CPU, dtype=torch.float64)
    fuel = fills[2]  # UO2 3%
    sw = _driver(_mesh(fuel), xs, **kw)
    sw.solve_eigenvalue(1e-11)
    assert sw.converged

    A = sw.assemble_dense_operator(0)
    disc = sw.get_discretization(0)
    mat = xs.get_material(fuel)
    D, N, G = sw.weights.numel(), disc.num_dofs, xs.num_groups
    chi_nuf = torch.outer(mat.chi, mat.nu_fission)  # (G, G'), rows destination
    # F[(d, v, g), (e, u, h)] = w_e chi_g nuSf_h M_vu: isotropic, so the
    # same for every outgoing direction d
    F = torch.einsum("e,vu,gh->vgeuh", sw.weights, disc.mass, chi_nuf)
    F = F.unsqueeze(0).expand(D, N, G, D, N, G).reshape(D * N * G, D * N * G)
    k_ref = torch.linalg.eigvals(torch.linalg.solve(A, F)).abs().max().item()
    assert abs(sw.k_eff - k_ref) < 1e-8


def test_inner_forcing_saves_sweeps():
    """Forced (inexact) inner solves reach the same k with fewer sweeps."""
    torch.set_default_dtype(torch.float64)
    fills, xs = kaist(problem="2B", device=CPU, dtype=torch.float64)
    runs = {}
    for forcing in (0.0, 0.1):
        sw = _driver(_mesh(fills[2]), xs, inner_forcing=forcing)
        sw.solve_eigenvalue(1e-11)
        assert sw.converged
        runs[forcing] = (sw.k_eff, sw.equivalent_sweeps)
    assert abs(runs[0.1][0] - runs[0.0][0]) < 1e-9
    assert runs[0.1][1] < runs[0.0][1]
