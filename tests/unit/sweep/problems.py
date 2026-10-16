"""Small multigroup, multi-patch problems in 1, 2 and 3 dimensions, with vacuum or
reflective boundaries, shared by the fiber-sweep and cross tests."""

import sys
from pathlib import Path

import torch
from igakit import cad

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.math import ProductQuadrature, QuadratureSet1D
from ttnte.mesh import IGAMesh
from ttnte.physics import BCPlane, BoundaryType, FixedSource
from ttnte.sweep import IGASweepDriver1D, IGASweepDriver2D, IGASweepDriver3D
from ttnte.xs import Material, Server

sys.path.insert(0, str(Path(__file__).parents[2] / "regression" / "single_patch"))
from test_eighth_sphere import create_solid_octant_sphere  # noqa: E402

CPU = torch.device("cpu")
DRIVERS = {1: IGASweepDriver1D, 2: IGASweepDriver2D, 3: IGASweepDriver3D}


def multigroup_xs(num_groups, fissile=False):
    """A scattering (optionally fissile) fuel and a moderator with downscatter and weak
    upscatter, scattering ratio below one."""
    G = num_groups
    g = torch.arange(G, dtype=torch.float64)
    materials = []
    server = Server()
    for name, base, ratio in [("Fuel", 1.0, 0.7), ("Moderator", 0.6, 0.9)]:
        m = Material(name)
        total = base * (1.0 + 0.15 * g)
        m.total = total
        scatter = torch.zeros(1, G, G, dtype=torch.float64)
        for src in range(G):
            out = ratio * total[src]
            for dst in range(G):
                if dst >= src:
                    scatter[0, src, dst] = 0.6 ** (dst - src)
                elif dst == src - 1:
                    scatter[0, src, dst] = 0.05
            scatter[0, src] *= out / scatter[0, src].sum()
        m.scatter_gtg = scatter
        if fissile and name == "Fuel":
            chi = torch.exp(-g)
            m.chi = chi / chi.sum()
            m.nu_fission = 0.4 * total
            m.fission = m.nu_fission / 2.5
            m.absorption = total - scatter[0].sum(1)
        m.finalize()
        server.add_material(m)
        materials.append(m)
    server.finalize()
    return materials, server


def quadrature(num_dim, n=2):
    if num_dim == 1:
        qset = QuadratureSet1D.gauss_legendre(2 * n)
    else:
        qset = ProductQuadrature.gauss_legendre_chebyshev(n, n, num_dim)
    qset.to_(CPU, torch.float64)
    return qset


def _source(num_groups):
    return FixedSource(
        isotropic_strength=torch.linspace(1.0, 0.2, num_groups, dtype=torch.float64)
    )


def _finish(mesh, num_dim, reflective):
    mesh.connect()
    if reflective:
        planes = [
            BCPlane(x_min=True),
            BCPlane(x_min=True, y_min=True),
            BCPlane(x_min=True, y_min=True, z_min=True),
        ][num_dim - 1]
        mesh.set_axis_aligned_conditions(planes, BoundaryType.REFLECTIVE, tol=1e-6)
    mesh.finalize()
    return mesh


def slab(num_groups, reflective=False, fissile=False):
    """Three patches on [0, 3]: fuel with a source, then moderator twice."""
    fills, xs = multigroup_xs(num_groups, fissile)
    mesh = IGAMesh(mpi_context)
    for k, (x0, x1) in enumerate([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]):
        patch = Patch.from_igakit(
            cad.refine(cad.line((x0, 0), (x1, 0)), 3, 2),
            device=CPU,
            dtype=torch.float64,
            fill=fills[0 if k == 0 else 1].label,
        )
        if k == 0 and not fissile:
            patch.source = _source(num_groups)
        mesh.add_block(patch)
    return _finish(mesh, 1, reflective), xs


def quarter_circle(num_groups, reflective=False, fissile=False, factor=2):
    """Four curved patches (the mc27 shielded quarter circle, multigroup)."""
    fills, xs = multigroup_xs(num_groups, fissile)
    c0 = cad.circle(radius=5, angle=torch.pi / 2)
    c1, c2 = c0.slice(0, 0, 0.5), c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))
    l1 = cad.line(p0=(6, 0), p1=(6, 6))
    l2 = cad.line(p0=(6, 6), p1=(0, 6))
    surfaces = [
        cad.ruled(l0, c1),
        cad.ruled(l0, c2),
        cad.ruled(c1, l1),
        cad.ruled(c2, l2),
    ]
    mesh = IGAMesh(mpi_context)
    for k, surface in enumerate(surfaces):
        patch = Patch.from_igakit(
            cad.refine(surface, factor, 2),
            device=CPU,
            dtype=torch.float64,
            fill=fills[0 if k < 2 else 1].label,
        )
        if k < 2 and not fissile:
            patch.source = _source(num_groups)
        mesh.add_block(patch)
    return _finish(mesh, 2, reflective), xs


def octant_sphere(num_groups, reflective=False, fissile=False, split=True):
    """A solid octant sphere, split into its Bezier elements (several units)."""
    fills, xs = multigroup_xs(num_groups, fissile)
    patch = Patch.from_igakit(
        cad.refine(create_solid_octant_sphere(2.0), 1, 2),
        device=CPU,
        dtype=torch.float64,
        fill=fills[0].label,
    )
    if not fissile:
        patch.source = _source(num_groups)
    mesh = IGAMesh(mpi_context)
    for block in patch.split_elements() if split else [patch]:
        mesh.add_block(block)
    return _finish(mesh, 3, reflective), xs


BUILDERS = {1: slab, 2: quarter_circle, 3: octant_sphere}


def driver(num_dim, mesh, xs):
    return DRIVERS[num_dim](mesh, xs, mpi_context)
