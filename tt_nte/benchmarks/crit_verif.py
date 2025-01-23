"""
crit_verif.py.

All cases below are given in
https://www.sciencedirect.com/science/article/pii/S0149197002000987.
All cases are given in a critical configuration.
"""

import gmsh
import numpy as np

from tt_nte.geometry import Geometry
from tt_nte.xs import Server


def _create_1d_mesh(model, materials, lengths, num_nodes, left_bc, right_bc):
    points = np.zeros(len(lengths) + 1, dtype=int)
    lines = np.zeros(len(lengths), dtype=int)

    # Start left BC point
    x = 0
    points[0] = model.geo.add_point(x, 0, 0)

    # Iterate through points and lines
    for i in range(len(materials)):
        x += lengths[i]
        points[i + 1] = model.geo.add_point(x, 0, 0)
        lines[i] = model.geo.add_line(points[i], points[i + 1])

    # Name materials
    for mat in np.unique(materials):
        model.add_physical_group(
            1, lines[np.argwhere(materials == mat).flatten()], name=mat
        )

    # Name boundary conditions
    if left_bc == right_bc:
        model.add_physical_group(0, [points[0], points[-1]], name=left_bc)
    else:
        model.add_physical_group(0, [points[0]], name=left_bc)
        model.add_physical_group(0, [points[-1]], name=right_bc)

    # Sync gmsh model
    model.geo.synchronize()

    # Create structured mesh
    for i in range(lines.size):
        # Transfinite curves
        model.mesh.set_transfinite_curve(lines[i], num_nodes[i])

    # Generate mesh
    model.mesh.generate(1)
    model.mesh.recombine()


def _create_2d_mesh(
    model, materials, lengths, x_num_nodes, y_num_nodes, left_bc, right_bc
):
    points = np.zeros((len(lengths) + 1, 2), dtype=int)
    x_lines = np.zeros(len(lengths) + 1, dtype=int)
    y_lines = np.zeros((len(lengths), 2), dtype=int)
    faces = np.zeros(len(lengths), dtype=int)
    surfaces = np.zeros(len(lengths), dtype=int)

    # Start left BC edge
    x = 0
    y = 0.1
    points[0, 0] = model.geo.add_point(x, 0, 0)
    points[0, 1] = model.geo.add_point(x, y, 0)
    x_lines[0] = gmsh.model.geo.add_line(points[0, 0], points[0, 1])

    # Iterate through points, lines, and surfaces
    for i in range(lengths.size):
        x += lengths[i]

        for j in range(2):
            points[i + 1, j] = model.geo.add_point(x, y * j, 0)
            y_lines[i, j] = model.geo.add_line(points[i, j], points[i + 1, j])

        x_lines[i + 1] = model.geo.add_line(points[i + 1, 0], points[i + 1, 1])

        faces[i] = model.geo.add_curve_loop(
            [-x_lines[i], y_lines[i, 0], x_lines[i + 1], -y_lines[i, 1]]
        )
        surfaces[i] = model.geo.add_plane_surface([faces[i]])

    # Name materials
    for mat in np.unique(materials):
        model.add_physical_group(
            2, surfaces[np.argwhere(materials == mat).flatten()], name=mat
        )

    # Name BCs
    vacuum = []
    reflective = y_lines.flatten().tolist()

    if left_bc == "reflective":
        reflective += [x_lines[0]]
    else:
        vacuum += [x_lines[0]]

    if right_bc == "reflective":
        reflective += [x_lines[-1]]
    else:
        vacuum += [x_lines[-1]]

    if vacuum:
        model.add_physical_group(1, vacuum, name="vacuum")
    model.add_physical_group(1, reflective, name="reflective")

    # Synch gmsh model
    model.geo.synchronize()

    # Create structured mesh
    model.mesh.set_transfinite_curve(x_lines[0], y_num_nodes)

    for i in range(lengths.size):
        # Transfinite curves
        for j in range(2):
            model.mesh.set_transfinite_curve(y_lines[i, j], x_num_nodes[i])

        model.mesh.set_transfinite_curve(x_lines[i + 1], y_num_nodes)

        # Transfinite surfaces
        model.mesh.set_transfinite_surface(
            surfaces[i], cornerTags=points[i : i + 1, :].flatten().tolist()
        )

    # Generate mesh
    model.mesh.generate(2)
    model.mesh.recombine()


def pu_brick(num_nodes, infinity=False):
    """
    Pu-239 1D slab problem taken from the Criticality Verification Benchmark Suite.

    The width of the slab is 3.707444 cm with vacuum boundary conditions on either side.
    """
    # Cross section data
    xs = {
        "chi": np.array([1.0]),
        "fuel": {
            "nu_fission": np.array([3.24 * 0.081600]),  # 1/cm
            "scatter_gtg": np.array([[[0.225216]]]),  # 1/cm
            "total": np.array([0.32640]),  # 1/cm
        },
    }

    # Slab Geometry
    gmsh.initialize()
    gmsh.model.add("Pu Brick")
    gmsh.option.setNumber("General.Terminal", 0)
    _create_1d_mesh(
        gmsh.model,
        np.array(["fuel"]),
        np.array([3.707444]),
        np.array([num_nodes]),
        "vacuum" if infinity is False else "reflective",
        "vacuum" if infinity is False else "reflective",
    )
    geometry = Geometry(gmsh.model)
    gmsh.finalize()

    return Server(xs), geometry


def pu_brick_multi_region(num_nodes, num_regions):
    """
    Same as pu_brick() but split into several Regions.

    Regions are linearly spaced.
    """
    # Cross section data
    xs_data = {
        "nu_fission": np.array([3.24 * 0.081600]),  # 1/cm
        "scatter_gtg": np.array([[[0.225216]]]),  # 1/cm
        "total": np.array([0.32640]),  # 1/cm
    }
    xs = {
        "chi": np.array([1.0]),
    }

    # Split regions in Server
    regions = [f"fuel_{i}" for i in range(num_regions)]
    for region in regions:
        xs[region] = xs_data

    xs_server = Server(xs)

    # Slab Geometry
    num_nodes = np.linspace(0, num_nodes, num_regions + 1, dtype=int)
    lengths = np.linspace(0, 3.707444, num_regions + 1)

    num_nodes = num_nodes[1:] - num_nodes[:-1]
    num_nodes[:-1] += 1
    lengths = lengths[1:] - lengths[:-1]

    gmsh.initialize()
    gmsh.model.add("Pu Brick (Multi-region)")
    gmsh.option.setNumber("General.Terminal", 0)
    _create_1d_mesh(
        gmsh.model,
        np.array(regions),
        lengths,
        num_nodes,
        "vacuum",
        "vacuum",
    )
    geometry = Geometry(gmsh.model)
    gmsh.finalize()

    return xs_server, geometry


def research_reactor_multi_region(num_nodes, right_bc="reflective", infinite=False):
    """
    Multi-region case with multiplying medium (fuel) and non-multiplying medium (mod).

    This case is also multi-group.
    """
    # Cross section data
    xs = {
        "chi": np.array([1.0, 0.0]),
        "fuel": {
            "nu_fission": 2.5 * np.array([0.000836, 0.029564]),  # 1/cm
            "scatter_gtg": np.array(
                [
                    [
                        [0.83892, 0.000767],
                        [0.04635, 2.918300],
                    ],
                ]
            ),  # 1/cm
            "total": np.array([0.88721, 2.9727]),  # 1/cm
        },
        "moderator": {
            "nu_fission": np.zeros(2),  # 1/cm
            "scatter_gtg": np.array(
                [
                    [
                        [0.83975, 0.000336],
                        [0.04749, 2.967600],
                    ],
                ]
            ),  # 1/cm
            "total": np.array([0.88798, 2.9865]),  # 1/cm
        },
    }
    xs_server = Server(xs)

    # Slab geometry
    regions = ["moderator", "fuel"]
    lengths = [1.126151, 6.696802]

    if infinite:
        right_bc = "reflective"

    if right_bc == "vacuum":
        regions += ["moderator"]
        lengths[-1] *= 2
        lengths += [lengths[0]]

    gmsh.initialize()
    gmsh.model.add("Multi-Region Research Reactor")
    gmsh.option.setNumber("General.Terminal", 0)
    _create_1d_mesh(
        gmsh.model,
        np.array(regions),
        np.array(lengths),
        np.array(num_nodes),
        "vacuum" if not infinite else "reflective",
        right_bc,
    )
    geometry = Geometry(gmsh.model)
    gmsh.finalize()

    return xs_server, geometry


def research_reactor_anisotropic(num_nodes, right_bc="reflective"):
    """Two-group uranium research reactor (linearly-anisotropic), Tables 49 and 50."""
    # Cross section data
    xs = {
        "chi": np.array([1.0, 0.0]),
        "fuel": {
            "total": np.array([0.65696, 2.52025]),  # 1/cm
            "nu_fission": 2.5 * np.array([0.0010484, 0.050632]),  # 1/cm
            "scatter_gtg": np.array(
                [
                    [
                        [0.625680, 0.00000],
                        [0.029227, 2.44383],
                    ],
                    [
                        [0.2745900, 0.00000],
                        [0.0075737, 0.83318],
                    ],
                ]
            ),  # 1/cm
        },
    }

    # Slab Geometry
    gmsh.initialize()
    gmsh.model.add("Multi-Region Research Reactor (Anisotropic)")
    gmsh.option.setNumber("General.Terminal", 0)
    _create_1d_mesh(
        gmsh.model,
        np.array(["fuel"]),
        np.array([9.4959 if right_bc == "reflective" else 2 * 9.4959]),
        np.array([num_nodes]),
        "vacuum",
        right_bc,
    )
    geometry = Geometry(gmsh.model)
    gmsh.finalize()

    return Server(xs), geometry


def pu_brick_2d(x_num_nodes, y_num_nodes, infinite=False):
    """
    Pu-239 1D slab problem taken from the Criticality Verification Benchmark Suite.

    The width of the slab is 3.707444 cm with vacuum boundary conditions on either side.
    """
    # Cross section data
    xs = {
        "chi": np.array([1.0]),
        "fuel": {
            "nu_fission": np.array([3.24 * 0.081600]),  # 1/cm
            "scatter_gtg": np.array([[[0.225216]]]),  # 1/cm
            "total": np.array([0.32640]),  # 1/cm
        },
    }

    # Slab Geometry
    gmsh.initialize()
    gmsh.model.add("Pu Brick (2D)")
    gmsh.option.setNumber("General.Terminal", 0)
    _create_2d_mesh(
        model=gmsh.model,
        materials=np.array(["fuel"]),
        lengths=np.array([3.707444]),
        x_num_nodes=np.array([x_num_nodes]),
        y_num_nodes=y_num_nodes,
        left_bc="vacuum" if infinite is False else "reflective",
        right_bc="vacuum" if infinite is False else "reflective",
    )
    geometry = Geometry(gmsh.model)
    gmsh.finalize()

    return Server(xs), geometry


def research_reactor_anisotropic_2d(x_num_nodes, y_num_nodes, right_bc="reflective"):
    """Two-group uranium research reactor (linearly-anisotropic), Tables 49 and 50."""
    # Cross section data
    xs = {
        "chi": np.array([1.0, 0.0]),
        "fuel": {
            "total": np.array([0.65696, 2.52025]),  # 1/cm
            "nu_fission": 2.5 * np.array([0.0010484, 0.050632]),  # 1/cm
            "scatter_gtg": np.array(
                [
                    [
                        [0.625680, 0.00000],
                        [0.029227, 2.44383],
                    ],
                    [
                        [0.2745900, 0.00000],
                        [0.0075737, 0.83318],
                    ],
                ]
            ),  # 1/cm
        },
    }

    # Slab Geometry
    gmsh.initialize()
    gmsh.model.add("Multi-Region Research Reactor (Anisotropic 2D)")
    gmsh.option.setNumber("General.Terminal", 0)
    _create_2d_mesh(
        gmsh.model,
        np.array(["fuel"]),
        np.array([9.4959 if right_bc == "reflective" else 2 * 9.4959]),
        np.array([x_num_nodes]),
        y_num_nodes,
        "vacuum",
        right_bc,
    )
    geometry = Geometry(gmsh.model)
    gmsh.finalize()

    return Server(xs), geometry


def research_reactor_multi_region_2d(x_num_nodes, y_num_nodes, right_bc="reflective"):
    """
    Multi-region case (in 2D) with multiplying medium (fuel) and non-multiplying medium
    (mod).

    This case is also multi-group.
    """
    # Cross section data
    xs = {
        "chi": np.array([1.0, 0.0]),
        "fuel": {
            "nu_fission": 2.5 * np.array([0.000836, 0.029564]),  # 1/cm
            "scatter_gtg": np.array(
                [
                    [
                        [0.83892, 0.000767],
                        [0.04635, 2.918300],
                    ],
                ]
            ),  # 1/cm
            "total": np.array([0.88721, 2.9727]),  # 1/cm
        },
        "moderator": {
            "nu_fission": np.zeros(2),  # 1/cm
            "scatter_gtg": np.array(
                [
                    [
                        [0.83975, 0.000336],
                        [0.04749, 2.967600],
                    ],
                ]
            ),  # 1/cm
            "total": np.array([0.88798, 2.9865]),  # 1/cm
        },
    }
    xs_server = Server(xs)

    # Slab geometry
    regions = ["moderator", "fuel"]
    lengths = [1.126151, 6.696802]

    if right_bc == "vacuum":
        regions += ["moderator"]
        lengths[-1] *= 2
        lengths += [lengths[0]]

    regions = np.array(regions)
    lengths = np.array(lengths)
    x_num_nodes = np.array(x_num_nodes)
    y_num_nodes = np.array(y_num_nodes)

    gmsh.initialize()
    gmsh.model.add("Multi-Region Research Reactor (2D)")
    gmsh.option.setNumber("General.Terminal", 0)
    _create_2d_mesh(
        model=gmsh.model,
        materials=regions,
        lengths=lengths,
        x_num_nodes=x_num_nodes,
        y_num_nodes=y_num_nodes,
        left_bc="vacuum",
        right_bc=right_bc,
    )
    geometry = Geometry(gmsh.model)
    gmsh.finalize()

    return xs_server, geometry
