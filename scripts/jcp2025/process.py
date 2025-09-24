import json
import os
import sys
import pickle
from pathlib import Path
from typing import Union, Tuple, Callable, List, Optional
import multiprocessing

import numpy as np
import torch as tn
from ttnte.assemblers import MatrixAssembler


def process(
    get_xs: Callable,
    get_mesh: Callable,
    dir: Union[str, Path],
    mc_leakage_frac: List[float],
    mc_solution: Optional[List[np.ndarray]] = None,
):
    # Make sure we have a path
    dir = Path(dir)

    # Get JSON (Lines) file
    jsonl_file = list(dir.rglob("*.jsonl"))[0]

    # Get regular mesh object
    regular_mesh = None

    i = 0
    with open(jsonl_file, "r", encoding="utf-8") as infile, open(
        jsonl_file.with_name(f"processed_{jsonl_file.name}"), "w"
    ) as outfile:
        # Iterate through lines
        for line in infile:
            # Parse JSON data
            result = json.loads(line)

            case = [
                result["num_ordinates"],
                result["num_groups"],
                (
                    result["factor"]
                    if isinstance(result["factor"], int)
                    else result["factor"][0]
                )
                + (
                    result["degree"]
                    if isinstance(result["degree"], int)
                    else result["degree"][0]
                ),
                (
                    result["factor"]
                    if isinstance(result["factor"], int)
                    else result["factor"][1]
                )
                + (
                    result["degree"]
                    if isinstance(result["degree"], int)
                    else result["degree"][1]
                ),
                (
                    result["degree"]
                    if isinstance(result["degree"], int)
                    else result["degree"][0]
                ),
                (
                    result["degree"]
                    if isinstance(result["degree"], int)
                    else result["degree"][1]
                ),
                result["eps"],
                result["device"],
            ]

            print(
                "Running case {}: N={}, G={}, A={}, B={}, p={}, q={}, eps={}, device={}".format(
                    i, *case
                )
            )
            i += 1

            # Get XSs and mesh
            xs_server = get_xs(result["num_groups"])
            mesh = get_mesh(result["factor"], result["degree"])

            # Map regular mesh if needed
            if regular_mesh == None and mc_solution is not None:
                regular_mesh = mesh.map_regular_mesh(
                    shape=mc_solution[0].shape[1:], N=(5, 5)
                )

            # Make a matrix assembler
            assembler = MatrixAssembler(
                mesh=mesh, xs_server=xs_server, num_ordinates=result["num_ordinates"]
            )

            # Read in solution data
            solution = pickle.load(
                open(
                    dir / "meshes/N{}_G{}_A{}_B{}_p{}_q{}_eps{}{}.pkl".format(*case),
                    "rb",
                )
            )
            # Calculate leakage fractions
            result["leakage_fraction"] = {
                "value": [],
                "error": [],
                "zscore": [],
            }

            if mc_solution is not None:
                result["flux_stats"] = {
                    "minimum": [],
                    "q1": [],
                    "median": [],
                    "q2": [],
                    "maximum": [],
                    "mean": [],
                    "l2 error": [],
                }

            for psi in solution.values():
                # Get leakage fraction data
                result["leakage_fraction"]["value"].append(
                    float(
                        assembler.outward_current(
                            tn.tensor(psi.reshape(assembler.discretization))
                        )
                        / assembler.total_production()
                    )
                )
                result["leakage_fraction"]["error"].append(
                    result["leakage_fraction"]["value"][-1] - mc_leakage_frac[0]
                )
                result["leakage_fraction"]["zscore"].append(
                    result["leakage_fraction"]["error"][-1] / mc_leakage_frac[1]
                )

                # Get stats for solution vector
                if mc_solution is not None:
                    assert regular_mesh is not None

                    # Get scalar flux
                    phi = assembler.angular_integral(
                        tn.tensor(psi.reshape(assembler.discretization))
                    ).reshape(assembler.discretization[-4:])

                    # Get phi average
                    error = np.empty(mc_solution[0].shape)

                    # Calculate stats comparing to MC solution
                    minimum = []
                    q1 = []
                    median = []
                    q2 = []
                    maximum = []
                    mean = []
                    l2 = []

                    for g in range(xs_server.num_groups):
                        # Set control points
                        mesh.set_phi(phi[g,])

                        # Set phi for mesh
                        error[g,] = (
                            mesh.regular_mesh(*regular_mesh) - mc_solution[0][g,]
                        )

                        z = error[g,] / mc_solution[1][g,]
                        minimum.append(np.min(z))
                        q1.append(np.percentile(z, 25))
                        median.append(np.median(z))
                        q2.append(np.percentile(z, 75))
                        maximum.append(np.max(z))
                        mean.append(np.mean(z))
                        l2.append(
                            np.linalg.norm(error[g,].flatten(), 2)
                            / np.linalg.norm(mc_solution[0][g,].flatten(), 2)
                        )

                    z = error / mc_solution[1]
                    minimum.append(np.min(z))
                    q1.append(np.percentile(z, 25))
                    median.append(np.median(z))
                    q2.append(np.percentile(z, 75))
                    maximum.append(np.max(z))
                    mean.append(np.mean(z))
                    l2.append(
                        np.linalg.norm(error.flatten(), 2)
                        / np.linalg.norm(mc_solution[0].flatten(), 2)
                    )

                    # Add stats to total result
                    result["flux_stats"]["minimum"].append(minimum)
                    result["flux_stats"]["q1"].append(q1)
                    result["flux_stats"]["median"].append(median)
                    result["flux_stats"]["q2"].append(q2)
                    result["flux_stats"]["maximum"].append(maximum)
                    result["flux_stats"]["mean"].append(mean)
                    result["flux_stats"]["l2 error"].append(l2)

            outfile.write(json.dumps(result) + "\n")
