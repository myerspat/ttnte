from pathlib import Path

import gmsh
import numpy as np
import pandas as pd

from tt_nte.geometry import Geometry
from tt_nte.xs import Server


def bwr_assembly(xy_num_nodes, control_rod=True, return_model=False):
    pitch = 1.3 * 1.2

    # Get number of nodes per cell
    num_elements_per_cell = np.ones(23, dtype=int) * int((xy_num_nodes / 23)) + 1
    num_elements_per_cell[: int(xy_num_nodes % 23 - 1)] += 1
    num_elements_per_cell = num_elements_per_cell[::-1]

    # Read XSs
    data = pd.read_csv(
        (
            Path(__file__).parent
            / "supporting/bwr/"
            / ("xs_assembly_control.dat" if control_rod else "xs_assembly.dat")
        ),
        delim_whitespace=True,
        header=None,
    ).values
    ordinates = pd.read_csv(
        (Path(__file__).parent / "./supporting/bwr/quadrature.dat"),
        delim_whitespace=True,
    ).values[:, [4, 1, 2]]
    ordinates[:, 0] = 0.25 * ordinates[:, 0] / (np.sum(ordinates[:, 0]))

    xs = {
        "chi": np.array(
            [
                1.0588870e-11,
                2.4979600e-11,
                1.9664890e-10,
                4.6093200e-10,
                8.2317503e-7,
                3.9163700e-4,
                0.13135319,
                0.86825800,
            ]
        )[::-1]
    }
    num_elements = int(np.max(data[:, 0]))
    num_groups = int(np.max(data[:, 1]))
    num_moments = int((data.shape[1] - 5) / num_groups)

    for i in range(num_elements):
        idx = np.argwhere(data[:, 0] == (i + 1)).flatten()
        scatter_gtg = np.zeros((num_moments, num_groups, num_groups))

        for l in range(num_moments):
            scatter_gtg[l, :, :] = data[
                idx, (5 + l * num_groups) : (5 + l * num_groups + num_groups)
            ].transpose()

        xs[str(i)] = {
            "total": data[idx, 2],
            "nu_fission": data[idx, 4],
            "scatter_gtg": scatter_gtg,
        }

    # Define assembly
    assembly = np.arange(num_elements).reshape(
        (int(np.sqrt(num_elements)), int(np.sqrt(num_elements)))
    )[:, ::-1]

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("BWR Core")
    gmsh.option.setNumber("General.Terminal", 0)

    # Create points between half unit cells
    points = np.zeros((assembly.shape[0] + 1, assembly.shape[1] + 1), dtype=int)
    for i in range(assembly.shape[0] + 1):
        for j in range(assembly.shape[1] + 1):
            points[i, j] = gmsh.model.geo.add_point(
                j * pitch / 2, i * pitch / 2, 0, 0.1, i * (assembly.shape[0] + 1) + j
            )

    # Connect points
    x_lines = np.zeros((assembly.shape[0], assembly.shape[1] + 1), dtype=int)
    y_lines = np.zeros((assembly.shape[0] + 1, assembly.shape[1]), dtype=int)

    for i in range(y_lines.shape[0]):
        for j in range(y_lines.shape[1]):
            y_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i, j + 1])

    for i in range(x_lines.shape[0]):
        for j in range(x_lines.shape[1]):
            x_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i + 1, j])

    # Add boundary conditions
    gmsh.model.add_physical_group(
        1,
        y_lines[-1, :].tolist()
        + x_lines[:, -1].tolist()
        + y_lines[0, :].tolist()
        + x_lines[:, 0].tolist(),
        name="reflective",
    )

    # Create surfaces
    faces = np.zeros(assembly.shape, dtype=int)
    surfaces = np.zeros(assembly.shape, dtype=int)

    for i in range(assembly.shape[0]):
        for j in range(assembly.shape[1]):
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
    for i in range(num_elements):
        idxs = np.argwhere(assembly == i)
        gmsh.model.add_physical_group(
            2, faces[idxs[:, 0], idxs[:, 1]].flatten().tolist(), name=str(i)
        )

    # Sync gmsh model
    gmsh.model.geo.synchronize()

    # Create structured mesh
    for i in range(assembly.shape[0]):
        for j in range(assembly.shape[1]):
            # Transfinite curves
            gmsh.model.mesh.set_transfinite_curve(
                y_lines[i, j], num_elements_per_cell[j]
            )
            gmsh.model.mesh.set_transfinite_curve(
                y_lines[i + 1, j], num_elements_per_cell[j]
            )
            gmsh.model.mesh.set_transfinite_curve(
                x_lines[i, j], num_elements_per_cell[i]
            )
            gmsh.model.mesh.set_transfinite_curve(
                x_lines[i, j + 1], num_elements_per_cell[i]
            )

            # Transfinite surface
            gmsh.model.mesh.set_transfinite_surface(
                surfaces[i, j],
                cornerTags=points[i : i + 2, j : j + 2].flatten().tolist(),
            )

    # Generate mesh and recombine to get 4-point quadrangle structured mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()

    if return_model:
        return Server(xs), Geometry(gmsh.model), ordinates, gmsh.model
    else:
        geometry = Geometry(gmsh.model)
        gmsh.finalize()

        return Server(xs), geometry, ordinates


def bwr_core(xy_num_nodes, reflected=False, return_model=False):
    diff = np.array([10] + 7 * [20] + [30])[::-1]

    # Get number of nodes per cell
    num_elements_per_cell = np.ones(9, dtype=int) * int((xy_num_nodes / 9)) + 1
    num_elements_per_cell[: int(xy_num_nodes % 9 - 1)] += 1
    num_elements_per_cell = num_elements_per_cell[::-1]

    # Read XSs
    data = pd.read_csv(
        (Path(__file__).parent / "supporting/bwr/xs_core.dat"),
        delim_whitespace=True,
        header=None,
    ).values
    ordinates = pd.read_csv(
        (Path(__file__).parent / "./supporting/bwr/quadrature.dat"),
        delim_whitespace=True,
    ).values[:, [4, 1, 2]]
    ordinates[:, 0] = 0.25 * ordinates[:, 0] / (np.sum(ordinates[:, 0]))

    num_groups = int(np.max(data[:, 1]))
    num_moments = int((data.shape[1] - 5) / num_groups)

    xs = {"chi": np.array([1.0, 0.0])}
    materials = ["fuel 1", "fuel 2", "fuel 3", "moderator"]

    for i in range(len(materials)):
        idx = np.argwhere(data[:, 0] == (i + 1)).flatten()
        scatter_gtg = np.zeros((num_moments, num_groups, num_groups))

        for l in range(num_moments):
            scatter_gtg[l, :, :] = data[
                idx, (5 + l * num_groups) : (5 + l * num_groups + num_groups)
            ].transpose()

        xs[materials[i]] = {
            "total": data[idx, 2],
            "nu_fission": data[idx, 3],
            "scatter_gtg": scatter_gtg,
        }

    # Define core layers
    core = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 5, 5, 5, 3, 3, 2, 2, 1],
            [12, 7, 12, 7, 7, 7, 2, 2, 1],
            [9, 13, 9, 13, 9, 11, 8, 4, 1],
            [12, 10, 12, 10, 16, 10, 8, 4, 1],
            [9, 13, 9, 13, 9, 15, 8, 6, 1],
            [12, 10, 12, 10, 14, 10, 14, 6, 1],
            [9, 13, 9, 13, 9, 15, 8, 6, 1],
            [12, 10, 12, 10, 14, 10, 14, 6, 1],
        ]
    )[:, ::-1]

    regions = {
        "0": "moderator",
        "1": "moderator",
        "2": "moderator",
        "3": "moderator",
        "4": "moderator",
        "5": "fuel 1",
        "6": "fuel 1",
        "7": "fuel 1",
        "8": "fuel 1",
        "9": "fuel 2",
        "10": "fuel 2",
        "11": "fuel 1",
        "12": "fuel 3",
        "13": "fuel 3",
        "14": "fuel 3",
        "15": "fuel 3",
        "16": "fuel 2",
    }

    if reflected:
        diff = np.concatenate([diff, diff[::-1]])
        num_elements_per_cell = np.concatenate(
            (num_elements_per_cell, num_elements_per_cell[::-1])
        )
        num_elements_per_cell[0] += 1
        core = np.block([[core, core[:, ::-1]], [core[::-1, :], core[::-1, ::-1]]])

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("BWR Core")
    gmsh.option.setNumber("General.Terminal", 0)

    # Create points
    points = np.zeros((core.shape[0] + 1, core.shape[1] + 1), dtype=int)
    x = 0
    y = 0
    for i in range(core.shape[0] + 1):
        for j in range(core.shape[1] + 1):
            points[i, j] = gmsh.model.geo.add_point(x, y, 0)

            if j != core.shape[1]:
                x += diff[j]
            else:
                x = 0

        if i != core.shape[0]:
            y += diff[i]
        else:
            y = 0

    # Connect points
    x_lines = np.zeros((core.shape[0], core.shape[1] + 1), dtype=int)
    y_lines = np.zeros((core.shape[0] + 1, core.shape[1]), dtype=int)

    for i in range(y_lines.shape[0]):
        for j in range(y_lines.shape[1]):
            y_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i, j + 1])

    for i in range(x_lines.shape[0]):
        for j in range(x_lines.shape[1]):
            x_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i + 1, j])

    # Add boundaries
    reflective = x_lines[:, -1].tolist() + y_lines[-1, :].tolist()
    vacuum = x_lines[:, 0].tolist() + y_lines[0, :].tolist()
    if reflected:
        gmsh.model.add_physical_group(1, reflective + vacuum, name="vacuum")
    else:
        gmsh.model.add_physical_group(1, vacuum, name="vacuum")
        gmsh.model.add_physical_group(1, reflective, name="reflective")

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
    for region in regions.keys():
        idxs = np.argwhere(core == int(region))
        gmsh.model.add_physical_group(
            2, faces[idxs[:, 0], idxs[:, 1]].flatten().tolist(), name=region
        )

    # Sync gmsh model
    gmsh.model.geo.synchronize()

    # Create structured mesh
    for i in range(core.shape[0]):
        for j in range(core.shape[1]):
            # Transfinite curves
            gmsh.model.mesh.set_transfinite_curve(
                y_lines[i, j], num_elements_per_cell[j]
            )
            gmsh.model.mesh.set_transfinite_curve(
                y_lines[i + 1, j], num_elements_per_cell[j]
            )
            gmsh.model.mesh.set_transfinite_curve(
                x_lines[i, j], num_elements_per_cell[i]
            )
            gmsh.model.mesh.set_transfinite_curve(
                x_lines[i, j + 1], num_elements_per_cell[i]
            )

            # Transfinite surface
            gmsh.model.mesh.set_transfinite_surface(
                surfaces[i, j],
                cornerTags=points[i : i + 2, j : j + 2].flatten().tolist(),
            )

    # Generate mesh and recombine to get 4-point quadrangle structured mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()

    if return_model:
        return Server(xs), Geometry(gmsh.model), ordinates, regions, gmsh.model
    else:
        geometry = Geometry(gmsh.model)
        gmsh.finalize()

        return Server(xs), geometry, ordinates, regions


def bwr_core_homogenized(xy_num_nodes, reflected=False, return_model=False):
    diff = np.array([10] + 7 * [20] + [30])[::-1]

    # Get number of nodes per cell
    num_elements_per_cell = np.array([xy_num_nodes])

    # Read XSs
    data = pd.read_csv(
        (Path(__file__).parent / "supporting/bwr/xs_core.dat"),
        delim_whitespace=True,
        header=None,
    ).values
    ordinates = pd.read_csv(
        (Path(__file__).parent / "./supporting/bwr/quadrature.dat"),
        delim_whitespace=True,
    ).values[:, [4, 1, 2]]
    ordinates[:, 0] = 0.25 * ordinates[:, 0] / (np.sum(ordinates[:, 0]))

    num_groups = int(np.max(data[:, 1]))
    num_moments = int((data.shape[1] - 5) / num_groups)

    # Define core layers
    core = np.array(
        [
            [3, 3, 3, 3, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 3, 3, 3, 3, 3],
            [2, 0, 2, 0, 0, 0, 3, 3, 3],
            [1, 2, 1, 2, 1, 0, 0, 3, 3],
            [2, 1, 2, 1, 1, 1, 0, 3, 3],
            [1, 2, 1, 2, 1, 2, 0, 0, 3],
            [2, 1, 2, 1, 2, 1, 2, 0, 3],
            [1, 2, 1, 2, 1, 2, 0, 0, 3],
            [2, 1, 2, 1, 2, 1, 2, 0, 3],
        ]
    )

    xs = {
        "chi": np.array([1.0, 0.0]),
        "0": {
            "total": np.zeros(num_groups),
            "nu_fission": np.zeros(num_groups),
            "scatter_gtg": np.zeros((num_moments, num_groups, num_groups)),
        },
    }

    for i in range(core.shape[0]):
        for j in range(core.shape[1]):
            idx = np.argwhere(data[:, 0] == core[i, j] + 1).flatten()
            scatter_gtg = np.zeros((num_moments, num_groups, num_groups))

            for l in range(num_moments):
                scatter_gtg[l, :, :] = data[
                    idx, (5 + l * num_groups) : (5 + l * num_groups + num_groups)
                ].transpose()

            xs["0"]["total"] += data[idx, 2] * diff[i] * diff[j]
            xs["0"]["nu_fission"] += data[idx, 3] * diff[i] * diff[j]
            xs["0"]["scatter_gtg"] += scatter_gtg * diff[i] * diff[j]

    xs["0"]["total"] /= np.sum(diff) ** 2
    xs["0"]["nu_fission"] /= np.sum(diff) ** 2
    xs["0"]["scatter_gtg"] /= np.sum(diff) ** 2
    core = np.array([[0]])
    diff = np.array([np.sum(diff)])

    if reflected:
        diff = diff * np.ones(2)
        num_elements_per_cell = num_elements_per_cell * np.ones(2, dtype=int)
        num_elements_per_cell[0] += 1
        core = core * np.ones((2, 2))

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("BWR Core")
    gmsh.option.setNumber("General.Terminal", 0)

    # Create points
    points = np.zeros((core.shape[0] + 1, core.shape[1] + 1), dtype=int)
    x = 0
    y = 0
    for i in range(core.shape[0] + 1):
        for j in range(core.shape[1] + 1):
            points[i, j] = gmsh.model.geo.add_point(x, y, 0)

            if j != core.shape[1]:
                x += diff[j]
            else:
                x = 0

        if i != core.shape[0]:
            y += diff[i]
        else:
            y = 0

    # Connect points
    x_lines = np.zeros((core.shape[0], core.shape[1] + 1), dtype=int)
    y_lines = np.zeros((core.shape[0] + 1, core.shape[1]), dtype=int)

    for i in range(y_lines.shape[0]):
        for j in range(y_lines.shape[1]):
            y_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i, j + 1])

    for i in range(x_lines.shape[0]):
        for j in range(x_lines.shape[1]):
            x_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i + 1, j])

    # Add boundaries
    reflective = x_lines[:, -1].tolist() + y_lines[-1, :].tolist()
    vacuum = x_lines[:, 0].tolist() + y_lines[0, :].tolist()
    if reflected:
        gmsh.model.add_physical_group(1, reflective + vacuum, name="vacuum")
    else:
        gmsh.model.add_physical_group(1, vacuum, name="vacuum")
        gmsh.model.add_physical_group(1, reflective, name="reflective")

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
    gmsh.model.add_physical_group(2, faces.flatten().tolist(), name="0")

    # Sync gmsh model
    gmsh.model.geo.synchronize()

    # Create structured mesh
    for i in range(core.shape[0]):
        for j in range(core.shape[1]):
            # Transfinite curves
            gmsh.model.mesh.set_transfinite_curve(
                y_lines[i, j], num_elements_per_cell[j]
            )
            gmsh.model.mesh.set_transfinite_curve(
                y_lines[i + 1, j], num_elements_per_cell[j]
            )
            gmsh.model.mesh.set_transfinite_curve(
                x_lines[i, j], num_elements_per_cell[i]
            )
            gmsh.model.mesh.set_transfinite_curve(
                x_lines[i, j + 1], num_elements_per_cell[i]
            )

            # Transfinite surface
            gmsh.model.mesh.set_transfinite_surface(
                surfaces[i, j],
                cornerTags=points[i : i + 2, j : j + 2].flatten().tolist(),
            )

    # Generate mesh and recombine to get 4-point quadrangle structured mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()

    if return_model:
        return Server(xs), Geometry(gmsh.model), ordinates, gmsh.model
    else:
        geometry = Geometry(gmsh.model)
        gmsh.finalize()

        return Server(xs), geometry, ordinates
