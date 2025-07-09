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
font = {"size": 10}
matplotlib.rc("font", **font)

if __name__ == "__main__":
    # Discretization
    num_ordinates = 1024

    # Get XS data
    xs_server = pu239(num_groups=1)

    # Create NURBS surface
    rc = 4.279960  # Critical radius (cm)
    c0 = cad.circle(radius=rc, angle=np.pi / 2)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))
    patches = {cad.ruled(l0, c0): "Pu-239"}

    # ===========================================
    # Plot initial mesh
    # Create mesh
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=3, degree=2)

    # Connect patches
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_condition(("left", "bottom"))

    # Finalize mesh
    mesh.finalize()

    # Plot final mesh
    ax = mesh.plot(meshlines=False)
    plt.tight_layout()
    plt.savefig("quarter_circle.png", dpi=300)

    # ===========================================
    # Plot scalar flux
    mesh = IGAMesh(patches)

    # Refine mesh resolution
    mesh.refine(0, factor=9, degree=3)

    # Connect patches
    mesh.connect()

    # Set reflective boundary conditions
    mesh.set_reflective_condition(("left", "bottom"))

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
    H, S, F, B_in, B_out = assembler.build(use_tt=True, eps=1e-10)

    # Save TT information
    assembler.save_tt_info("./tt_info.csv")

    # Solve
    k, psi = eig(
        LHS=LinearOperator([H, B_out, -S, -B_in], N=assembler.N, M=assembler.M),
        RHS=LinearOperator([F], N=assembler.N, M=assembler.M),
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

    # Calculate eigenvalue error
    print("keff error: {} pcm".format((k - 1) * 1e5))

    def evaluate_boundary(patch):
        # Calculate flux in the center
        center_flux = patch.evaluate_single((0, 0))[-1]

        # Evaluate boundary flux
        points = np.ones((2, 400))
        points[0, :] = np.linspace(0, 1, 400)
        points = np.array(patch.evaluate_list(points.copy().T.tolist()))

        # Normalize boundary
        points[:, -1] /= center_flux

        # Convert to angle
        angular_points = np.zeros((points.shape[0], 2))
        angular_points[:, -1] = points[:, -1]
        angular_points[:, 0] = np.arcsin(points[(points[:, 0] >= 0), 1] / rc)
        return angular_points[angular_points[:, 0].argsort()]

    def evaluate_radius(mesh, patch, radius, tol=1e-10):
        # Plot and evaluate boundary flux
        center_flux = patch.evaluate_single((0, 0))[-1]

        # Calculate physical locations
        points = radius * np.ones((2, 400))
        angular_points = np.zeros((2, 400))
        angular_points[0, :] = np.linspace(0, np.pi / 2, 400)
        points[0, :] *= np.cos(angular_points[0, :])
        points[1, :] *= np.sin(angular_points[0, :])

        # Inverse map to the parametric domain
        _, coords = mesh.inverse_map(points, tol=tol)
        pred = np.array(patch.evaluate_list(coords.T.tolist())).T
        angular_points[1, :] = pred[-1, :] / center_flux

        return angular_points

    # Change control points for phi
    mesh.set_phi(phi)
    patch = mesh.patches[0]

    # Plot scalar flux solution at rc
    points = evaluate_boundary(patch)
    print(points[:, 1])
    print(
        np.linalg.norm(points[:, 1] - 0.2926, 2)
        / np.sqrt(np.sum((0.2926 * np.ones(points[:, 1].size)) ** 2))
    )
    plt.clf()
    plt.plot(points[:, 0], points[:, 1])
    plt.hlines(0.2926, 0, np.pi / 2, color="black", linestyles="--")
    plt.legend(["Calculated", "Benchmark Solution"])
    plt.ylabel("$\\phi(r = r_c) / \\phi(r = 0)$")
    plt.xlabel("Polar Angle")
    plt.tight_layout()
    plt.savefig("./r_1.png", dpi=300)

    # Plot scalar flux solution at 0.5rc
    points = evaluate_radius(mesh, patch, radius=0.5 * rc)
    print(
        np.linalg.norm(points[1, :] - 0.8093, 2)
        / np.sqrt(np.sum((0.8093 * np.ones(points[1, :].size)) ** 2))
    )
    plt.clf()
    plt.plot(points[0, :], points[1, :])
    plt.hlines(0.8093, 0, np.pi / 2, color="black", linestyles="--")
    plt.ylabel("$\\phi(r = 0.5r_c) / \\phi(r = 0)$")
    plt.xlabel("Polar Angle")
    plt.legend(["Calculated", "Benchmark Solution"])
    plt.tight_layout()
    plt.savefig("./r_0.5.png", dpi=300)

    # Plot scalar flux
    plt.clf()
    ax = mesh.plot(plot_ctrlpts=False)
    ax.set_zlabel(f"$\\phi_1" + "(\\hat{x}, \\hat{y})$")
    ax.view_init(18, -18, 0)
    plt.tight_layout()
    plt.savefig("./phi_1.png", dpi=300)
