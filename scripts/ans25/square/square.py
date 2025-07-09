import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from igakit import cad

from ttnte.assemblers import TTAssembler
from ttnte.iga import IGAMesh
from ttnte.linalg import LinearOperator, eig
from ttnte.xs.benchmarks import pu239

tn.set_default_dtype(tn.float64)
font = {"size": 14}
matplotlib.rc("font", **font)

if __name__ == "__main__":
    # Discretization
    num_nodes = 256
    num_ordinates = 1024

    # Get XS data
    xs_server = pu239(num_groups=2)

    # Create NURBS geometry
    length = 6.5  # cm
    points = np.array(
        [
            [-length / 2, -length / 2, 0],
            [length / 2, -length / 2, 0],
            [-length / 2, length / 2, 0],
            [length / 2, length / 2, 0],
        ]
    ).reshape((2, 2, -1))
    patches = {cad.bilinear(points): "Pu-239"}

    # ===========================================
    # Plot initial mesh
    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=3, degree=2)

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()

    # Plot final mesh
    ax = mesh.plot(meshlines=False)
    plt.tight_layout()
    plt.savefig("square.png", dpi=300)

    # ===========================================
    # Plot scalar flux
    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=16, degree=3)

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()

    # Create operators in TT format
    assembler = TTAssembler(
        mesh=mesh,
        xs_server=xs_server,
        num_ordinates=num_ordinates,
    )
    assembler.interp_jacobian = False
    assembler.interp_jacobian_det = False
    assembler.interp_boundary_jacobian_det = False
    H, S, F, _, B_out = assembler.build(use_tt=True, eps=1e-10)

    # Save TT information
    assembler.save_tt_info("./tt_info.csv")

    # Run solver
    k, psi = eig(
        LHS=LinearOperator([H, B_out, -S], N=assembler.N, M=assembler.M),
        RHS=LinearOperator([F], N=assembler.N, M=assembler.M),
        tols=1e-8,
        max_iters=500,
        device=0,
    )

    # Compute scalar flux
    phi = (
        assembler.angular_integral(psi)
        .numpy()
        .reshape((xs_server.num_groups, mesh.num_patches, -1))
    )

    # Get OpenMC solution
    k_mc = [0.997955, 0.000031]
    phi_mc = np.load("../../../notebooks/square/openmc/data/mesh_flux.npy")

    # Ensure OpenMC solution is normalized
    phi_mc /= np.linalg.norm(phi_mc.flatten(), 2)

    # Calculate eigenvalue error
    print("keff error: {} +/- {} pcm".format((k - k_mc[0]) * 1e5, k_mc[1]))

    # Map rectangular mesh
    pids, coords = mesh.map_regular_mesh(shape=phi_mc.shape[1:], N=(2, 2))

    # Iterate through groups and plot
    phi_avg = np.zeros(phi_mc.shape)
    for g in range(xs_server.num_groups):
        # Set control points
        mesh.set_phi(phi[g,])

        # Calculate regular mesh
        phi_avg[g,] = mesh.regular_mesh(pids, coords)

        # Plot
        ax = mesh.plot(plot_ctrlpts=False)
        ax.set_zlabel(f"$\\phi_{g + 1}" + "(\\hat{x}, \\hat{y})$")
        plt.tight_layout()
        plt.savefig(f"./phi_{g + 1}.png", dpi=300)

    # Normalize average
    phi_avg /= np.linalg.norm(phi_avg.flatten(), 2)

    for g in range(xs_server.num_groups):
        # Calculate groups L2-error
        print(
            "Scalar flux Relative L2-error (g = {}): {}".format(
                g + 1,
                np.linalg.norm((phi_avg[g,] - phi_mc[g,]).flatten(), 2)
                / np.linalg.norm(phi_mc[g,].flatten(), 2),
            )
        )
    print(
        "Total scalar flux Relative L2-error: {}".format(
            np.linalg.norm((phi_avg - phi_mc).flatten(), 2)
            / np.linalg.norm(phi_mc.flatten(), 2)
        )
    )
