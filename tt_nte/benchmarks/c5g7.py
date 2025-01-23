"""
c5g7.py.

C5G7 benchmark with variable spatial MGXS on pincell level. MGXS were homogenized to
conform to a regular grid using OpenMC. Therefore the MGXS tensor is 7x51x51.
"""

import pickle
from typing import Optional
from pathlib import Path

import gmsh
import numpy as np

from tt_nte.xs import Server
from tt_nte.geometry import Geometry


def c5g7_2d(
    num_cells_pin: int,
    num_cells_mod: Optional[int] = None,
    reflective_bc: bool = True,
    return_model: bool = False,
):
    """
    Load 2D C5G7 benchmark.

    C5G7 benchmark with variable spatial MGXS on pincell level. MGXS were
    homogenized to conform to a regular grid using OpenMC. Therefore the MGXS
    tensor is 7x51x51.

    Parameters
    ==========
    num_cells_pin: int
        Number of spatial cells along a given dimension within the pins of
        the assemblies.
    num_cells_mod: int, optional
        Number of spatial cells along a given dimension in the moderator
        region. If ``num_cells_mod`` is not specified then the default is
        ``num_cells_mod = 17 * num_cells_pin``.
    reflective_bc: bool, default=True
        Whether to have a reflective boundary condition along the top and
        left boundaries.
    return_model: bool, default=False
        Whether to return the Gmsh model.

    Returns
    =======
    geometry: tt_nte.geometry.Geometry
        Geometry object with given discretization.
    server: tt_nte.xs.Server
        MGXS server for raw data.
    model: gmsh.model
        Gmsh model. Only returned if ``return_model = True``.
    """
    # Load MGXS data
    with open(Path(__file__).parent / "supporting/c5g7/xs.pkl", "rb") as file:
        xs_server = Server(pickle.load(file))

    # Materials
    w = 0  # Water pin
    u = 1  # UO2 fuel pin
    g = 2  # Guide tupe pin
    f = 3  # Fission chamber pin
    m = 4  # 4.3% MOX fuel pin
    n = 5  # 7% MOX fuel pin
    o = 6  # 8.7% MOX fuel pin

    mats = {
        "moderator": w,
        "uo2": u,
        "guide_tube": g,
        "fission_chamber": f,
        "4.3 mox": m,
        "7 mox": n,
        "8.7 mox": o,
    }

    # Assemblies
    uo2_assembly = np.array(
        [
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
            [u, u, u, u, u, g, u, u, g, u, u, g, u, u, u, u, u],
            [u, u, u, g, u, u, u, u, u, u, u, u, u, g, u, u, u],
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
            [u, u, g, u, u, g, u, u, g, u, u, g, u, u, g, u, u],
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
            [u, u, g, u, u, g, u, u, f, u, u, g, u, u, g, u, u],
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
            [u, u, g, u, u, g, u, u, g, u, u, g, u, u, g, u, u],
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
            [u, u, u, g, u, u, u, u, u, u, u, u, u, g, u, u, u],
            [u, u, u, u, u, g, u, u, g, u, u, g, u, u, u, u, u],
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
            [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        ]
    )
    mox_assembly = np.array(
        [
            [m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m],
            [m, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, m],
            [m, n, n, n, n, g, n, n, g, n, n, g, n, n, n, n, m],
            [m, n, n, g, n, o, o, o, o, o, o, o, n, g, n, n, m],
            [m, n, n, n, o, o, o, o, o, o, o, o, o, n, n, n, m],
            [m, n, g, o, o, g, o, o, g, o, o, g, o, o, g, n, m],
            [m, n, n, o, o, o, o, o, o, o, o, o, o, o, n, n, m],
            [m, n, n, o, o, o, o, o, o, o, o, o, o, o, n, n, m],
            [m, n, g, o, o, g, o, o, f, o, o, g, o, o, g, n, m],
            [m, n, n, o, o, o, o, o, o, o, o, o, o, o, n, n, m],
            [m, n, n, o, o, o, o, o, o, o, o, o, o, o, n, n, m],
            [m, n, g, o, o, g, o, o, g, o, o, g, o, o, g, n, m],
            [m, n, n, n, o, o, o, o, o, o, o, o, o, n, n, n, m],
            [m, n, n, g, n, o, o, o, o, o, o, o, n, g, n, n, m],
            [m, n, n, n, n, g, n, n, g, n, n, g, n, n, n, n, m],
            [m, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, m],
            [m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m],
        ]
    )

    # Define C5G7 core
    core = np.block(
        [
            2 * [w * np.ones((1, uo2_assembly.shape[1]))] + [np.array([[w]])],
            [mox_assembly, uo2_assembly, w * np.ones((uo2_assembly.shape[0], 1))],
            [uo2_assembly, mox_assembly, w * np.ones((uo2_assembly.shape[0], 1))],
        ]
    )

    # Pin pitch and number of cells in moderator
    num_cells_mod = num_cells_mod if num_cells_mod else 17 * num_cells_pin
    pin_pitch = 1.26  # cm

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("C5G7")
    gmsh.option.setNumber("General.Terminal", 0)

    # Create base points defining pin cells and water region
    points = np.zeros(
        (int(2 * uo2_assembly.shape[0] + 2), int(2 * uo2_assembly.shape[1] + 2)),
        dtype=int,
    )
    for j in range(points.shape[1] - 1):
        points[0, j] = gmsh.model.geo.add_point(j * pin_pitch, 0, 0)
    points[0, -1] = gmsh.model.geo.add_point(64.26, 0, 0)

    for i in range(1, points.shape[0]):
        for j in range(points.shape[1] - 1):
            points[i, j] = gmsh.model.geo.add_point(
                j * pin_pitch, 21.42 + i * pin_pitch, 0
            )
        points[i, -1] = gmsh.model.geo.add_point(64.26, 21.42 + i * pin_pitch, 0)

    # Connect points
    # x_lines: all lines with constant x position
    # y_lines: all lines with constant y position
    x_lines = np.zeros((points.shape[0] - 1, points.shape[1]), dtype=int)
    y_lines = np.zeros((points.shape[0], points.shape[1] - 1), dtype=int)

    for i in range(y_lines.shape[0]):
        for j in range(y_lines.shape[1]):
            y_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i, j + 1])

    for i in range(x_lines.shape[0]):
        for j in range(x_lines.shape[1]):
            x_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i + 1, j])

    # Add boundary conditions
    if reflective_bc:
        gmsh.model.add_physical_group(
            1, y_lines[0, :].tolist() + x_lines[:, -1].tolist(), name="vacuum"
        )
        gmsh.model.add_physical_group(
            1, y_lines[-1, :].tolist() + x_lines[:, 0].tolist(), name="reflective"
        )

    else:
        gmsh.model.add_physical_group(
            1,
            y_lines[0, :].tolist()
            + x_lines[:, -1].tolist()
            + y_lines[-1, :].tolist()
            + x_lines[:, 0].tolist(),
            name="vacuum",
        )

    # Create surfaces
    faces = np.zeros(core.shape, dtype=int)
    surfaces = np.zeros(core.shape, dtype=int)

    for i in range(core.shape[0]):
        for j in range(core.shape[1]):
            faces[i, j] = gmsh.model.geo.add_curve_loop(
                [
                    y_lines[i, j],
                    x_lines[i, j + 1],
                    -y_lines[i + 1, j],
                    -x_lines[i, j],
                ]
            )
            surfaces[i, j] = gmsh.model.geo.add_plane_surface([faces[i, j]])

    # Assign material regions
    for mat, id in mats.items():
        gmsh.model.add_physical_group(2, faces[core == id].flatten().tolist(), name=mat)

    # Sync gmsh model
    gmsh.model.geo.synchronize()

    # Create structured mesh
    for i in range(core.shape[0]):
        for j in range(core.shape[1]):
            # Determine number of cells along x and y
            y_num_nodes = (num_cells_pin if i > 0 else num_cells_mod) + 1
            x_num_nodes = (
                num_cells_pin if j < core.shape[1] - 1 else num_cells_mod
            ) + 1

            # Transfinite curves
            gmsh.model.mesh.set_transfinite_curve(y_lines[i, j], x_num_nodes)
            gmsh.model.mesh.set_transfinite_curve(y_lines[i + 1, j], x_num_nodes)
            gmsh.model.mesh.set_transfinite_curve(x_lines[i, j], y_num_nodes)
            gmsh.model.mesh.set_transfinite_curve(x_lines[i, j + 1], y_num_nodes)

            # Transfinite surface
            gmsh.model.mesh.set_transfinite_surface(
                surfaces[i, j],
                cornerTags=points[i : i + 2, j : j + 2].flatten().tolist(),
            )

    # Generate mesh and recombine to get 4-point quadrangle structured mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()

    if return_model:
        return xs_server, Geometry(gmsh.model), gmsh.model
    else:
        geometry = Geometry(gmsh.model)
        gmsh.finalize()

        return xs_server, geometry
