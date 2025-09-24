import sys
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    sys.path.append("../..")

import numpy as np

from square import get_xs, get_mesh
from process import process

if __name__ == "__main__":
    # Get MC solutions
    leakage_frac_openmc = [0.42095701399999963, 2.2038687252709062e-05]
    phi_mc = np.load(
        "../../../../notebooks/fixed_source/square/openmc/data/mesh_flux.npy"
    )
    phi_mc_stdev = np.load(
        "../../../../notebooks/fixed_source/square/openmc/data/mesh_stdev.npy"
    )

    # Process results
    process(
        get_xs, get_mesh, "./direction", leakage_frac_openmc, [phi_mc, phi_mc_stdev]
    )

    # Process results
    process(get_xs, get_mesh, "./meshsize", leakage_frac_openmc, [phi_mc, phi_mc_stdev])
