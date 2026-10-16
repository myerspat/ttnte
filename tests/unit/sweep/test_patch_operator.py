"""The sweeper discretizes exactly the system the TT assembler builds: the
dense patch operator (streaming + collision + outflow - reflective inflow -
scattering) agrees with the TT interior operator to round-off."""

import sys
from pathlib import Path

import pytest
import torch
from igakit.cad import circle, coons, line, refine

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.driver import (
    IGATransportDriver1D,
    IGATransportDriver2D,
    IGATransportDriver3D,
)
from ttnte.math import ProductQuadrature, QuadratureSet1D
from ttnte.mesh import IGAMesh
from ttnte.physics import BCPlane, BoundaryType, DGTransportAssemblerConfig
from ttnte.sweep import (
    IGASweepDriver1D,
    IGASweepDriver2D,
    IGASweepDriver3D,
    SweepConfig,
)
from ttnte.xs.benchmarks import pu239

# The octant-sphere geometry lives with the single-patch regressions, whose
# modules import their helpers relative to that directory.
sys.path.insert(0, str(Path(__file__).parents[2] / "regression" / "single_patch"))
from test_eighth_sphere import create_solid_octant_sphere  # noqa: E402

CPU = torch.device("cpu")


def _config():
    cfg = DGTransportAssemblerConfig()
    cfg.rounding.eps = 1e-14
    cfg.cross.eps = 1e-14
    cfg.max_dense_size = int(1e10)
    cfg.cross_jacobian_inverse = False
    return cfg


def _mesh(surface, fill, planes):
    mesh = IGAMesh(mpi_context)
    mesh.add_block(
        Patch.from_igakit(surface, device=CPU, dtype=torch.float64, fill=fill)
    )
    mesh.connect()
    if planes is not None:
        mesh.set_axis_aligned_conditions(planes, BoundaryType.REFLECTIVE, tol=1e-9)
    mesh.finalize()
    return mesh


def _compare(make_mesh, qset, tt_driver, sweep_driver, xs):
    tt = tt_driver(make_mesh(), xs, mpi_context)
    tt.assemble(qset, _config())
    A = tt.get_system(0).interior_op.as_tt()
    n = 1
    for m in A.m_modes:
        n *= m
    A = A.to_dense().reshape(n, n)

    sw = sweep_driver(make_mesh(), xs, mpi_context)
    sw.setup(qset, SweepConfig(verbose=False))
    B = sw.assemble_dense_operator(0)
    assert B.shape == A.shape
    assert ((A - B).norm() / A.norm()).item() < 1e-11


@pytest.mark.parametrize("fill_idx", [0, 1])
def test_operator_2d_curved_reflective(fill_idx):
    torch.set_default_dtype(torch.float64)
    fills, xs = pu239(num_groups=1, device=CPU, dtype=torch.float64)
    r = 3.0
    c0 = circle(r, angle=torch.pi / 4)
    c1 = circle(r, angle=-torch.pi / 4).rotate(torch.pi / 2)
    surf = coons([[line((0, 0), (0, r)), c0], [line((0, 0), (r, 0)), c1]])
    qset = ProductQuadrature.gauss_legendre_chebyshev(2, 2, 2)
    qset.to_(CPU, torch.float64)
    _compare(
        lambda: _mesh(
            refine(surf, 2, 2), fills[fill_idx], BCPlane(x_min=True, y_min=True)
        ),
        qset,
        IGATransportDriver2D,
        IGASweepDriver2D,
        xs,
    )


def test_operator_1d_slab():
    torch.set_default_dtype(torch.float64)
    fills, xs = pu239(num_groups=1, device=CPU, dtype=torch.float64)
    qset = QuadratureSet1D.gauss_legendre(8)
    qset.to_(CPU, torch.float64)
    _compare(
        lambda: _mesh(
            refine(line((0, 0), (2.0, 0)), 4, 2), fills[0], BCPlane(x_min=True)
        ),
        qset,
        IGATransportDriver1D,
        IGASweepDriver1D,
        xs,
    )


def test_operator_3d_octant_sphere():
    torch.set_default_dtype(torch.float64)
    fills, xs = pu239(num_groups=1, device=CPU, dtype=torch.float64)
    qset = ProductQuadrature.gauss_legendre_chebyshev(2, 2, 3)
    qset.to_(CPU, torch.float64)
    _compare(
        lambda: _mesh(
            refine(create_solid_octant_sphere(2.0), 1, 2),
            fills[0],
            BCPlane(x_min=True, y_min=True, z_min=True),
        ),
        qset,
        IGATransportDriver3D,
        IGASweepDriver3D,
        xs,
    )
