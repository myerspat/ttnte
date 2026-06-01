import os
from itertools import product

import pytest
import torch
import numpy as np
from igakit import cad

from ttnte import mpi_context
from ttnte.xs import Material, Server
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import BoundaryType, BCPlane

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    # ("cuda", torch.float32),
    # ("cuda", torch.float64),
]


def create_xs_server(device, dtype):
    # Source material
    source = Material("source")
    source.total = torch.tensor([1.0], device=device, dtype=dtype)
    source.scatter_gtg = torch.tensor([[[1.0 * 0.9]]], device=device, dtype=dtype)
    source.finalize()

    # Void material
    void = Material("void")
    void.total = torch.tensor([0.0], device=device, dtype=dtype)
    void.scatter_gtg = torch.tensor([[[0.0]]], device=device, dtype=dtype)
    void.finalize()

    # Create server and add materials
    server = Server()
    server.add_material(source)
    server.add_material(void)
    server.finalize()

    return [source.label, void.label], server


def create_coarse_patches(device, dtype):
    # Create quarter circle NURBS surface
    radius = 5  # cm
    pitch = 12  # cm
    c0 = cad.circle(radius=radius, angle=np.pi / 2)
    c1 = c0.slice(0, 0, 0.5)
    c2 = c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))

    # Create water patch
    l1 = cad.line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l2 = cad.line(p0=(pitch / 2, pitch / 2), p1=(0, pitch / 2))

    # Create igakit volumes
    source = [cad.ruled(l0, c1), cad.ruled(l0, c2)]
    void = [cad.ruled(c1, l1), cad.ruled(c2, l2)]

    # Extrude all surface
    zdelta = 10  # cm
    source = [cad.extrude(s, displ=zdelta, axis=2) for s in source]
    void = [cad.extrude(s, displ=zdelta, axis=2) for s in void]

    # Refine so there is no hanging nodes
    source = [cad.refine(s, 9, 3) for s in source]
    void = [cad.refine(v, 9, 3) for v in void]

    # Convert to ttnte patches
    source = [Patch.from_igakit(s, device=device, dtype=dtype) for s in source]
    void = [Patch.from_igakit(s, device=device, dtype=dtype) for s in void]

    return source, void


@pytest.mark.parametrize("device, dtype", test_params)
def test_mesh(device, dtype):
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    torch.set_num_threads(int(os.cpu_count() / mpi_context.world_size))

    # Get XS info and patches
    (slabel, vlabel), _ = create_xs_server(device, dtype)
    spatches, vpatches = create_coarse_patches(device, dtype)

    # Set the material type for the patches
    for s, v in zip(spatches, vpatches):
        s.fill = slabel
        v.fill = vlabel

    # Create a mesh and add these to the mesh
    mesh = IGAMesh(0)
    mesh.reserve(4)
    for p in spatches + vpatches:
        mesh.add_block(p)

    # Mark internal interfaces and connect them
    mesh.connect()

    # Expected connections
    expected_connections = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
    expected_unknowns = {0: 3, 1: 3, 2: 4, 3: 4}
    expected_vacuums = {0: 0, 1: 0, 2: 0, 3: 0}
    expected_degeneracies = {0: 1, 1: 1, 2: 0, 3: 0}

    connections = {0: [], 1: [], 2: [], 3: []}
    unknowns = {0: 0, 1: 0, 2: 0, 3: 0}
    vacuums = {0: 0, 1: 0, 2: 0, 3: 0}
    degeneracies = {0: 0, 1: 0, 2: 0, 3: 0}

    for i, patch in enumerate(mesh.blocks):
        for dim, is_upper in product([0, 1, 2], [False, True]):
            boundary = patch.get_boundary_info(dim, is_upper)
            assert boundary.fid == dim * 2 + is_upper

            # Check boundary types
            if boundary.type == BoundaryType.INTERNAL:
                assert len(boundary.connections) == 1
                connections[i] = sorted(connections[i] + [boundary.connections[0].gid])
            elif boundary.type == BoundaryType.UNKNOWN:
                unknowns[i] += 1
            elif boundary.type == BoundaryType.VACUUM:
                vacuums[i] += 1
            elif boundary.type == BoundaryType.DEGENERATE:
                degeneracies[i] += 1
            else:
                raise RuntimeError("Unexpected boundary type found")

    assert expected_connections == connections
    assert expected_unknowns == unknowns
    assert expected_vacuums == vacuums
    assert expected_degeneracies == degeneracies

    # Test the mapping method for mesh
    face_a = mesh.blocks[0].get_boundary(0, 0)
    face_b = Patch()
    face_b.ctrlptsw = torch.flip(face_a.ctrlptsw.permute((1, 0, 2)), (0,)).contiguous()
    face_b.basis = face_a.basis[::-1]
    is_coupled, mapping_a, mapping_b = mesh.get_boundary_mapping(face_a, face_b)

    assert is_coupled == True
    assert mapping_a.flip == [False, True]
    assert mapping_a.perm == [1, 0, 2]
    assert mapping_b.flip == [True, False]
    assert mapping_b.perm == [1, 0, 2]

    # Set reflective boundary conditions
    mesh.set_axis_aligned_conditions(
        BCPlane(z_min=True, z_max=True), BoundaryType.REFLECTIVE, tol=1e-5
    )

    expected_reflectives = {0: 2, 1: 2, 2: 2, 3: 2}
    expected_unknowns = {0: 1, 1: 1, 2: 2, 3: 2}
    expected_degeneracies = {0: 1, 1: 1, 2: 0, 3: 0}

    connections = {0: [], 1: [], 2: [], 3: []}
    reflectives = {0: 0, 1: 0, 2: 0, 3: 0}
    unknowns = {0: 0, 1: 0, 2: 0, 3: 0}
    vacuums = {0: 0, 1: 0, 2: 0, 3: 0}
    degeneracies = {0: 0, 1: 0, 2: 0, 3: 0}
    for i, patch in enumerate(mesh.blocks):
        for dim, is_upper in product([0, 1, 2], [False, True]):
            boundary = patch.get_boundary_info(dim, is_upper)

            # Check boundary types
            if boundary.type == BoundaryType.INTERNAL:
                assert len(boundary.connections) == 1
                connections[i] = sorted(connections[i] + [boundary.connections[0].gid])
            elif boundary.type == BoundaryType.UNKNOWN:
                unknowns[i] += 1
            elif boundary.type == BoundaryType.VACUUM:
                vacuums[i] += 1
            elif boundary.type == BoundaryType.REFLECTIVE:
                reflectives[i] += 1
            elif boundary.type == BoundaryType.DEGENERATE:
                degeneracies[i] += 1
            else:
                raise RuntimeError("Unexpected boundary type found")

    assert expected_connections == connections
    assert expected_unknowns == unknowns
    assert expected_vacuums == vacuums
    assert expected_reflectives == reflectives
    assert expected_degeneracies == degeneracies

    # Finalize the mesh and check
    mesh.finalize()

    expected_unknowns = {0: 0, 1: 0, 2: 0, 3: 0}
    expected_vacuums = {0: 1, 1: 1, 2: 2, 3: 2}
    expected_degeneracies = {0: 1, 1: 1, 2: 0, 3: 0}

    connections = {0: [], 1: [], 2: [], 3: []}
    reflectives = {0: 0, 1: 0, 2: 0, 3: 0}
    unknowns = {0: 0, 1: 0, 2: 0, 3: 0}
    vacuums = {0: 0, 1: 0, 2: 0, 3: 0}
    degeneracies = {0: 0, 1: 0, 2: 0, 3: 0}
    for i, patch in enumerate(mesh.blocks):
        for dim, is_upper in product([0, 1, 2], [False, True]):
            boundary = patch.get_boundary_info(dim, is_upper)

            # Check boundary types
            if boundary.type == BoundaryType.INTERNAL:
                assert len(boundary.connections) == 1
                connections[i] = sorted(connections[i] + [boundary.connections[0].gid])
            elif boundary.type == BoundaryType.UNKNOWN:
                unknowns[i] += 1
            elif boundary.type == BoundaryType.VACUUM:
                vacuums[i] += 1
            elif boundary.type == BoundaryType.REFLECTIVE:
                reflectives[i] += 1
            elif boundary.type == BoundaryType.DEGENERATE:
                degeneracies[i] += 1
            else:
                raise RuntimeError("Unexpected boundary type found")

    assert expected_connections == connections
    assert expected_unknowns == unknowns
    assert expected_vacuums == vacuums
    assert expected_reflectives == reflectives
    assert expected_degeneracies == degeneracies

    # Check the connectivity graph
    conn_graph = mesh.build_connectivity_graph()
    torch.testing.assert_close(
        conn_graph.local_gids,
        torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cpu"),
    )
    torch.testing.assert_close(
        conn_graph.xadj,
        torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64, device="cpu"),
    )
    torch.testing.assert_close(
        conn_graph.adjncy,
        torch.tensor([1, 2, 0, 3, 3, 0, 2, 1], dtype=torch.int64, device="cpu"),
    )
    torch.testing.assert_close(
        conn_graph.mpi_ranks,
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cpu"),
    )
