import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from igakit import cad

from ttnte.assemblers import MatrixAssembler, TTAssembler
from ttnte.iga import IGAMesh
from ttnte.linalg import LinearOperator, eig
from ttnte.xs.benchmarks import c5g7

tn.set_default_dtype(tn.float64)
font = {"size": 10}
matplotlib.rc("font", **font)

if __name__ == "__main__":
    # Discretization
    num_ordinates = 1024

    # Get XS data
    xs_server = c5g7()

    # Create quarter circle NURBS surface
    radius = 0.54  # cm
    pitch = 1.26  # cm
    c0 = cad.circle(radius=radius, angle=np.pi / 2)
    c1 = c0.slice(0, 0, 0.5)
    c2 = c0.slice(0, 0.5, 1)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))
    fuel1 = cad.ruled(l0, c1)
    fuel2 = cad.ruled(l0, c2)

    # Create water patch
    l1 = cad.line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l2 = cad.line(p0=(pitch / 2, pitch / 2), p1=(0, pitch / 2))
    mod1 = cad.ruled(c1, l1)
    mod2 = cad.ruled(c2, l2)

    # NURBS surfaces
    patches = {}
    patches[fuel1] = "UO2"
    patches[fuel2] = "UO2"
    patches[mod1] = "Water"
    patches[mod2] = "Water"

    # ===========================================
    # Plot initial mesh
    # Create IGA mesh object
    mesh = IGAMesh(patches)

    # Refine mesh
    for p in range(mesh.num_patches):
        mesh.refine(p, 3, 2)

    # Finalize mesh
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_condition(("left", "bottom", "top", "right"))

    # Finalize mesh
    mesh.finalize()

    # Plot final mesh
    ax = mesh.plot(meshlines=False)
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig("pincell.png", dpi=300)

    # ===========================================
    # Plot scalar flux
    # Create IGA mesh object
    mesh = IGAMesh(patches)

    # Refine mesh
    for p in range(mesh.num_patches):
        mesh.refine(p, 6, 2)

    # Finalize mesh
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_condition(("left", "bottom", "top", "right"))

    # Finalize mesh
    mesh.finalize()

    print("Assemble system in COOrdinate format")
    assembler = MatrixAssembler(
        mesh=mesh,
        xs_server=xs_server,
        num_ordinates=num_ordinates,
    )
    H_m, S_m, F_m, B_in_m, B_out_m = assembler.build()

    # Create operators in TT format
    print("\nAssemble system in TT format")
    assembler = TTAssembler(
        mesh=mesh,
        xs_server=xs_server,
        num_ordinates=num_ordinates,
    )
    assembler.interp_jacobian = False
    assembler.interp_jacobian_det = False
    assembler.interp_boundary_jacobian_det = False
    H_tt, S_tt, F_tt, B_in_tt, B_out_tt = assembler.build(use_tt=True, eps=1e-10)

    # Save TT information
    assembler.save_tt_info("./tt_info.csv")

    k, psi = eig(
        LHS=LinearOperator(
            [H_tt, B_out_m - B_in_m, -S_tt], N=assembler.N, M=assembler.M
        ),
        RHS=LinearOperator([F_tt], N=assembler.N, M=assembler.M),
        tols=1e-8,
        max_iters=500,
        device=0,
        linear_solver_opts={
            "max_iterations": 100,
            "threshold": 1e-10,
            "resets": 10,
        },
    )

    # Compute scalar flux
    phi = (
        assembler.angular_integral(psi)
        .numpy()
        .reshape((xs_server.num_groups, mesh.num_patches, -1))
    )

    # Get OpenMC solution
    k_mc = [1.325593, 0.000032]
    phi_mc = np.load("../../../notebooks/pincell/openmc/data/mesh_flux.npy")

    # Ensure OpenMC solution is normalized
    phi_mc /= np.linalg.norm(phi_mc.flatten(), 2)

    # Calculate eigenvalue error
    print("keff error: {} +/- {} pcm".format((k - k_mc[0]) * 1e5, k_mc[1]))

    # Map rectangular mesh
    pids, coords = mesh.map_regular_mesh(shape=phi_mc.shape[1:], N=(2, 2))

    # Iterate through groups and plot
    phi_avg = np.zeros(phi_mc.shape)
    elev = []
    azim = []
    for g in range(xs_server.num_groups):
        # Set control points
        mesh.set_phi(phi[g,])

        # Calculate regular mesh
        phi_avg[g,] = mesh.regular_mesh(pids, coords)

        # Plot
        plt.clf()
        ax = mesh.plot(plot_ctrlpts=False)
        ax.set_zlabel(f"$\\phi_{g + 1}" + "(\\hat{x}, \\hat{y})$")
        plt.tight_layout()
        plt.show()
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
