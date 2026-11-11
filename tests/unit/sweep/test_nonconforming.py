"""Non-conforming patch interfaces (one surface refined differently on its two sides,
whole face to whole face): the mesh connects them, and the sweep integrates both sides
of every interface with one shared quadrature, so particles are conserved to round-
off."""

import math

import pytest
import torch
from igakit import cad

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import BoundaryType, FixedSource
from ttnte.sweep import LocalSolverType, PatchDiscretization, SweepConfig
from ttnte.xs import Material, Server

from .problems import CPU, DRIVERS, quadrature

G = 2


def absorber():
    """Two purely absorbing materials (no scattering), so every patch's balance involves
    only its source, absorption and face fluxes."""
    server = Server()
    fills = []
    for name, sigma in [("A", 1.0), ("B", 0.5)]:
        m = Material(name)
        m.total = sigma * (1.0 + 0.2 * torch.arange(G, dtype=torch.float64))
        m.scatter_gtg = torch.zeros(1, G, G, dtype=torch.float64)
        m.finalize()
        server.add_material(m)
        fills.append(m)
    server.finalize()
    return fills, server


def build(surfaces, fills):
    mesh = IGAMesh(mpi_context)
    for k, nrb in enumerate(surfaces):
        patch = Patch.from_igakit(
            nrb, device=CPU, dtype=torch.float64, fill=fills[k % 2].label
        )
        if k == 0:
            patch.source = FixedSource(
                isotropic_strength=torch.linspace(1.0, 0.5, G, dtype=torch.float64)
            )
        mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()
    return mesh


def squares(conforming):
    """[0, 1]^2 and [1, 2] x [0, 1] with 3 and 4 (non-nested) spans along the shared
    edge."""
    left = cad.ruled(cad.line((0, 0), (0, 1)), cad.line((1, 0), (1, 1)))
    right = cad.ruled(cad.line((1, 0), (1, 1)), cad.line((2, 0), (2, 1)))
    n = 3 if conforming else 4
    return [
        cad.refine(left, factor=(3, 2), degree=2),
        cad.refine(right, factor=(n, 2), degree=2),
    ]


def rings(conforming, flip=False):
    """A quarter annulus split at r = 2: 3 angular spans inside, 5 outside (curved,
    rational interface), the outer ring optionally parametrized backwards."""
    inner = cad.ruled(
        cad.circle(radius=1, angle=math.pi / 2), cad.circle(radius=2, angle=math.pi / 2)
    )
    outer = cad.ruled(
        cad.circle(radius=2, angle=math.pi / 2), cad.circle(radius=3, angle=math.pi / 2)
    )
    inner = cad.refine(inner, factor=(5 if conforming else 3, 2), degree=2)
    outer = cad.refine(outer, factor=(5, 2), degree=2)
    if flip:
        outer = outer.reverse(0)
    return [inner, outer]


def cylinder_rings(conforming):
    """The rings extruded in z with different angular and axial spans on the two sides
    of the cylindrical interface (as in the Big Ten benchmark)."""
    surfaces = rings(True)
    out = []
    for k, s in enumerate(surfaces):
        v = cad.extrude(s, 1.0, 2)
        if conforming:
            out.append(cad.refine(v, factor=(4, 2, 2), degree=2))
        else:
            out.append(cad.refine(v, factor=(2 + 2 * k, 2, 2 + k), degree=2))
    return out


def solve(mesh, xs, num_dim, solver=LocalSolverType.BANDED_LU):
    sw = DRIVERS[num_dim](mesh, xs, mpi_context)
    sw.setup(
        quadrature(num_dim),
        SweepConfig(
            tol=1e-12,
            inner_tol=1e-13,
            store_angular_flux=True,
            local_solver=solver,
            verbose=False,
        ),
    )
    sw.solve_fixed_source()
    return sw


def balance_residual(sw, mesh, fills):
    """Per (direction, group): source - absorption - vacuum leakage, summed over all
    patches, relative to the source. Interface fluxes cancel exactly only if the
    outflow of one side equals the inflow of the other."""
    omega = sw.ordinates
    D = omega.shape[0]
    Y0 = sw.harmonics[:, 0]
    q = torch.linspace(1.0, 0.5, G, dtype=torch.float64)
    total = torch.zeros(D, G, dtype=torch.float64)
    source = None
    for k, block in enumerate(mesh.blocks):
        disc = sw.get_discretization(block.gid)
        psi = sw.get_angular_flux(block.gid).reshape(D, disc.num_dofs, G)
        integral = disc.basis_integral
        sigma = fills[k % 2].total
        total -= torch.einsum("v,dvg->dg", integral, psi) * sigma
        if k == 0:
            source = Y0[:, None] * q[None, :] * integral.sum()
            total += source
        for dim in range(disc.ndim):
            for up in (False, True):
                if block.get_boundary_info(dim, up).type == BoundaryType.INTERNAL:
                    continue
                face = disc.face(dim, up)
                c = torch.clamp(omega @ face.normal.T, min=0.0) * face.weights
                vals = c[:, face.pair_q] * face.pair_val
                total -= torch.einsum(
                    "dk,dkg->dg", vals, psi[:, face.dofs[face.pair_b], :]
                )
    return (total.abs().max() / source.abs().max()).item()


CASES = {
    "squares": (2, squares, {}),
    "rings": (2, rings, {}),
    "rings_flipped": (2, rings, {"flip": True}),
    "cylinder_rings": (3, cylinder_rings, {}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_connections_are_flagged(case):
    num_dim, builder, kw = CASES[case]
    fills, _ = absorber()
    for conforming in (True, False):
        mesh = build(builder(conforming, **kw), [f for f in fills])
        conns = [
            c
            for b in mesh.blocks
            for d in range(num_dim)
            for up in (False, True)
            for c in b.get_boundary_info(d, up).connections
        ]
        assert len(conns) == 2
        assert all(c.conforming == conforming for c in conns)


@pytest.mark.parametrize("case", list(CASES))
def test_particle_balance(case):
    num_dim, builder, kw = CASES[case]
    fills, xs = absorber()
    for conforming in (True, False):
        mesh = build(builder(conforming, **kw), fills)
        sw = solve(mesh, xs, num_dim)
        assert balance_residual(sw, mesh, fills) < 1e-11


@pytest.mark.parametrize("case", ["rings", "rings_flipped", "cylinder_rings"])
def test_close_to_conforming(case):
    """The non-conforming solution is a coarser discretization of the same problem:
    its absorption stays within discretization error of the conforming one."""
    num_dim, builder, kw = CASES[case]
    fills, xs = absorber()
    absorbed = []
    for conforming in (True, False):
        mesh = build(builder(conforming, **kw), fills)
        sw = solve(mesh, xs, num_dim)
        a = 0.0
        for k, block in enumerate(mesh.blocks):
            disc = sw.get_discretization(block.gid)
            phi = sw.get_scalar_flux(block.gid).reshape(disc.num_dofs, G)
            a += float((disc.basis_integral @ phi * fills[k % 2].total).sum())
        absorbed.append(a)
    assert abs(absorbed[0] - absorbed[1]) / absorbed[0] < 2e-2


def test_native_quadrature_would_not_conserve():
    """The check above is sensitive: on the curved interface, the outflow of the
    inner ring integrated with its own (native) face quadrature differs from the
    shared-quadrature outflow its neighbor receives."""
    fills, xs = absorber()
    mesh = build(rings(False), fills)
    sw = solve(mesh, xs, 2)
    block = mesh.blocks[0]
    shared = sw.get_discretization(block.gid)
    native = PatchDiscretization(block)
    omega = sw.ordinates
    psi = sw.get_angular_flux(block.gid).reshape(omega.shape[0], shared.num_dofs, G)
    for dim in range(2):
        for up in (False, True):
            if block.get_boundary_info(dim, up).type != BoundaryType.INTERNAL:
                continue
            out = []
            for disc in (shared, native):
                face = disc.face(dim, up)
                c = torch.clamp(omega @ face.normal.T, min=0.0) * face.weights
                vals = c[:, face.pair_q] * face.pair_val
                out.append(
                    torch.einsum("dk,dkg->dg", vals, psi[:, face.dofs[face.pair_b], :])
                )
            assert ((out[0] - out[1]).abs().max() / out[0].abs().max()).item() > 1e-8


def test_band_ordering_narrows_band_and_matches_dense():
    """The largest parametric dimension varies slowest in the band storage; the banded
    and dense LU sweeps agree."""
    fills, xs = absorber()

    def tall():
        return [
            cad.refine(cad.extrude(s, 1.0, 2), factor=(2, 2, 6), degree=2)
            for s in rings(True)
        ]

    mesh = build(tall(), fills)
    sw = solve(mesh, xs, 3)
    disc = sw.get_discretization(mesh.blocks[0].gid)
    nz = disc.shape[2]
    assert disc.band_order is not None
    # z moves slowest: band row 1 is the next r at z = 0 (natural index nz)
    assert disc.band_order[1].item() == nz
    assert disc.kl < disc.shape[0] * disc.shape[1] * 3
    dense = solve(build(tall(), fills), xs, 3, LocalSolverType.DENSE_LU)
    for block in mesh.blocks:
        a = sw.get_angular_flux(block.gid)
        b = dense.get_angular_flux(block.gid)
        assert ((a - b).norm() / b.norm()).item() < 1e-11
