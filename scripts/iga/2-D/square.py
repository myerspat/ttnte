import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from igakit import cad
from mesh import IGAMesh
from operator_builder import OperatorBuilder
from solve import eig

from tt_nte.xs import Server

# Set default precision
tn.set_default_dtype(tn.float64)
tn.set_num_threads(14)


def _volume_average_cell(xl, xr, yl, yr, patch, I0=5):
    x = np.linspace(xl, xr, I0)[1:-1]
    y = np.linspace(yl, yr, I0)[1:-1]

    return (
        1
        / 4
        * (
            patch.evaluate_single((xl, yl))[-1]
            + patch.evaluate_single((xl, yr))[-1]
            + patch.evaluate_single((xr, yr))[-1]
            + patch.evaluate_single((xr, yl))[-1]
            + 2 * sum(patch.evaluate_single((x[i], yl))[-1] for i in range(x.size))
            + 2 * sum(patch.evaluate_single((x[i], yr))[-1] for i in range(x.size))
            + 2 * sum(patch.evaluate_single((xl, y[i]))[-1] for i in range(y.size))
            + 2 * sum(patch.evaluate_single((xr, y[i]))[-1] for i in range(y.size))
            + 4
            * sum(
                sum(patch.evaluate_single((x[i], y[j]))[-1] for i in range(x.size))
                for j in range(y.size)
            )
        )
    )


if __name__ == "__main__":
    # Discretization
    num_nodes = 256
    num_ordinates = 1024

    # Settings
    tol = 1e-8  # Fission source convergence
    eps = 1e-10  # TT compression threshold

    # =================================================================
    # 2-group Pu Brick (1-Patch Square System)
    # =================================================================
    # Sides of square
    length = 6.5  # cm

    # Cross section data
    xs = {
        "chi": np.array([0.575, 0.425]),
        "fuel": {
            "total": np.array([0.2208, 0.3360]),
            "nu_fission": np.array([3.1 * 0.0936, 2.93 * 0.08544]),
            "scatter_gtg": np.array([[[0.0792, 0], [0.0432, 0.23616]]]),
        },
    }
    xs_server = Server(xs)

    # Create NURBS geometry
    patches = {}
    points = np.array(
        [
            [-length / 2, -length / 2, 0],
            [length / 2, -length / 2, 0],
            [-length / 2, length / 2, 0],
            [length / 2, length / 2, 0],
        ]
    ).reshape((2, 2, -1))
    patch = cad.bilinear(points)
    patches[patch] = "fuel"

    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=8, degree=4)

    # Finalize patches
    mesh.finalize_patches()

    # Define boundary conditions
    mesh.set_boundary_condition(0, 1, 0, "vacuum")  # Bottom
    mesh.set_boundary_condition(0, 1, 1, "vacuum")  # Top
    mesh.set_boundary_condition(0, 0, 0, "vacuum")  # Left
    mesh.set_boundary_condition(0, 0, 1, "vacuum")  # Right

    # Connect patches
    mesh.connect()

    # Plot final mesh
    ax = mesh.plot()
    ax.view_init(90, -90, 0)
    plt.tight_layout()
    plt.savefig("./figs/square/square.png", dpi=300)

    # Create operators
    builder = OperatorBuilder(
        mesh=mesh,
        xs_server=xs_server,
        num_ordinates=num_ordinates,
        fixed_source=False,
    )
    builder.interp_jacobian = False
    builder.interp_jacobian_det = False
    builder.interp_boundary_jacobian_det = False
    H, S, F = builder.build()

    # Run IGA solver
    k_iga, phi_iga = eig(
        H=H,
        S=S,
        F=F,
        Int_N=builder.Int_N,
        Int_I=builder.Int_I,
        tol=tol,
        eps=eps,
        max_fission_iter=500,
        max_scatter_iter=20,
        # sparsity_frac=0,
    )
    phi_iga = phi_iga.reshape((2, -1))

    # Get OpenMC solution
    k_mc = 0.997951
    phi_mc = np.load("./flux.npy")
    phi_mc /= np.linalg.norm(phi_mc.flatten(), 2)

    # Calculate eigenvalue error
    print("keff error: {} pcm".format((k_iga - k_mc) * 1e5))

    # Iterate through groups
    phi = np.zeros(phi_mc.shape)

    # Get X and Y
    x = np.linspace(0, 1, 257)
    y = np.linspace(0, 1, 257)
    Xl, Yl = np.meshgrid(x[:-1], y[:-1])
    Xr, Yr = np.meshgrid(x[1:], y[1:])
    patch = mesh.patches[0]
    for g in range(2):
        # Change control points for phi
        new_ctrlpts = patch.ctrlpts
        for i in range(phi_iga.shape[-1]):
            new_ctrlpts[i] = [*new_ctrlpts[i][:-1], phi_iga[g, i]]
        patch.ctrlpts = new_ctrlpts

        volume_average = np.vectorize(_volume_average_cell, otypes=[float])
        phi[g,] = volume_average(Xl, Xr, Yl, Yr, patch).reshape((256, 256))

        # Plot result
        plt.clf()
        mesh.plot()
        plt.tight_layout()
        plt.savefig(f"./figs/square/phi_{g + 1}.png", dpi=300)

    # Normalize
    phi /= np.linalg.norm(phi.flatten(), 2)

    # Calculate L2-error
    print(
        "Total L2 Error: {}".format(
            np.linalg.norm((phi - phi_mc).flatten(), 2)
            / np.linalg.norm(phi_mc.flatten(), 2)
        )
    )

    # Visualize
    for g in range(xs_server.num_groups):
        # Calculate L2 Error for each group
        print(
            "Group {} L2 Error: {}".format(
                g + 1,
                np.linalg.norm((phi[g,] - phi_mc[g,]).flatten(), 2)
                / np.linalg.norm(phi_mc[g,].flatten(), 2),
            )
        )
