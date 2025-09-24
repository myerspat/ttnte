import os
import sys
import json
import pickle
import itertools
import multiprocessing
from pathlib import Path
from typing import Union, Tuple

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from extract import get_jsonl_data, get_pickle_data


if __name__ == "__main__":
    # Path to this directory
    dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Solutions from OpenMC
    leakage_frac_openmc = [0.42095701399999963, 2.2038687252709062e-05]
    phi_mc = np.load(
        "../../../../notebooks/fixed_source/square/openmc/data/mesh_flux.npy"
    )
    phi_mc_stdev = np.load(
        "../../../../notebooks/fixed_source/square/openmc/data/mesh_stdev.npy"
    )

    num_ordinates = [16, 64, 256, 1024, 4096, 16384, 65536]
    degrees = [2, 3, 4, 6]
    eps = [1e-8, 1e-5, 1e-3]

    # Extract leakage fraction data
    data = get_jsonl_data(
        dir / "direction/processed_direction.jsonl",
        lambda line_data: (
            (True, line_data["leakage_fraction"])
            if line_data["device"] == "cpu"
            else (False, None)
        ),
    )

    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]

    # Plot leakage fraction
    for i, degree in enumerate(degrees):
        plt.plot(
            num_ordinates,
            [d["value"] for d in data if d["eps"] == eps[0] and d["degree"] == degree],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"CSR: $p = {degree}$",
        )
    plt.hlines(
        [leakage_frac_openmc[0]],
        num_ordinates[0],
        num_ordinates[-1],
        label="OpenMC$\\pm 2\\sigma$",
        color="black",
    )
    plt.fill_between(
        num_ordinates,
        -2 * leakage_frac_openmc[-1],
        2 * leakage_frac_openmc[-1],
        color="black",
        alpha=0.2,
    )
    plt.ylabel("Leakage Fraction")
    plt.xlabel("Number of Ordinates")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot CSR leakage fraction Z-score to OpenMC
    for i, degree in enumerate(degrees):
        plt.plot(
            num_ordinates,
            [d["zscore"] for d in data if d["eps"] == eps[0] and d["degree"] == degree],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"CSR: $p = {degree}$",
        )
    plt.ylabel("$#$ of $\\sigma$ from OpenMC")
    plt.xlabel("Number of Ordinates")
    plt.title("Leakage Fraction Compared to OpenMC")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot CSR leakage fraction error to OpenMC
    plt.clf()
    for i, degree in enumerate(degrees):
        plt.plot(
            num_ordinates,
            [d["error"] for d in data if d["eps"] == eps[0] and d["degree"] == degree],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"CSR: $p = {degree}$",
        )
    plt.ylabel("Leakage Fraction Error")
    plt.xlabel("Number of Ordinates")
    plt.title("Leakage Fraction Compared to OpenMC")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # for i, d in enumerate(degrees):
    #     csr = np.array([v["value"][0] for v in arrays[(d, eps[0])]])
    #     for j, e in enumerate(eps):
    #         el = ""
    #         if e == 1e-8:
    #             el = "10^{8}"
    #         if e == 1e-5:
    #             el = "10^{5}"
    #         if e == 1e-3:
    #             el = "10^{3}"
    #
    #         plt.plot(
    #             num_ordinates,
    #             (
    #                 np.array(
    #                     [v["value"][j if j != 0 else j + 1] for v in arrays[(d, e)]]
    #                 )
    #                 - csr
    #             )
    #             / csr,
    #             label=f"$\\epsilon={el}$",
    #             marker=markers[i],
    #             linestyle=line_styles[i],
    #         )
    #
    #     plt.ylabel("Leakage Fraction Error")
    #     plt.xlabel("Number of Ordinates")
    #     plt.legend()
    #     plt.xscale("log")
    #     plt.yscale("log")
    #     plt.show()

    # for i, d in enumerate(degrees):
    #     for j, e in enumerate(eps):
    #         plt.plot(
    #             num_ordinates,
    #             (
    #                 [v["error"][j - 1] for v in arrays[(d, e)]]
    #                 if j != 0
    #                 else [v["error"][j] for v in arrays[(d, 1e-8)]]
    #             ),
    #             label=(f"TT: $p={d},\\epsilon={e}$" if j != 0 else f"CSR: $p={d}$"),
    #             linestyle=line_styles[j],
    #             color=colors[j],
    #             marker=markers[i],
    #         )

    # plt.ylabel("Leakage Fraction Error")
    # plt.xlabel("Number of Ordinates")
    # plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
