import matplotlib.pyplot as plt
import numpy as np
import torch as tn
from igakit.nurbs import NURBS
from mesh import IGAMesh
from operator_builder import OperatorBuilder
from solve import eig

from tt_nte.xs import Server

# Set default precision
tn.set_default_dtype(tn.float64)
tn.set_num_threads(14)


def circle(radius):
    """"""
    # Control points and weights
    ctrlpts = radius * np.array(
        [
            [-1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, -np.sqrt(2), 0],
            [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [-np.sqrt(2), 0, 0],
            [0, 0, 0],
            [np.sqrt(2), 0, 0],
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, np.sqrt(2), 0],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        ]
    ).reshape((3, 3, 3))
    weights = np.array(
        [
            [1, 1 / np.sqrt(2), 1],
            [1 / np.sqrt(2), 1, 1 / np.sqrt(2)],
            [1, 1 / np.sqrt(2), 1],
        ]
    )

    return NURBS(
        knots=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], control=ctrlpts, weights=weights
    )


def evaluate_boundary(patch):
    # Plot and evaluate boundary flux
    center_flux = patch.evaluate_single((0.5, 0.5))[-1]
    points = np.zeros((2, 400))
    points[0, :100] = np.linspace(0, 1, 100)
    points[0, 100:200] = 1
    points[1, 100:200] = np.linspace(0, 1, 100)
    points[0, 200:300] = np.linspace(0, 1, 100)[::-1]
    points[1, 200:300] = 1
    points[1, 300:400] = np.linspace(0, 1, 100)[::-1]

    points = np.array(patch.evaluate_list(points.T.tolist()))
    points[:, -1] /= center_flux

    # Convert to angle
    angular_points = np.zeros((points.shape[0], 2))
    angular_points[:, -1] = points[:, -1]
    angular_points[(points[:, 0] >= 0), 0] = np.arcsin(
        points[(points[:, 0] >= 0), 1] / rc
    )
    angular_points[(points[:, 0] < 0) & (points[:, 1] >= 0), 0] = (
        -np.arcsin(points[(points[:, 0] < 0) & (points[:, 1] >= 0), 1] / rc) + np.pi
    )
    angular_points[(points[:, 0] < 0) & (points[:, 1] < 0), 0] = (
        -np.arcsin(points[(points[:, 0] < 0) & (points[:, 1] < 0), 1] / rc) - np.pi
    )
    return angular_points[angular_points[:, 0].argsort()]


def evaluate_radius(mesh, patch, radius, tol=1e-8):
    # Plot and evaluate boundary flux
    center_flux = patch.evaluate_single((0.5, 0.5))[-1]

    # Calculate physical locations
    points = radius * np.ones((2, 400))
    angular_points = np.zeros((2, 400))
    angular_points[0, :] = np.linspace(-np.pi, np.pi, 400)
    points[0, :] *= np.cos(angular_points[0, :])
    points[1, :] *= np.sin(angular_points[0, :])

    # Inverse map to the parametric domain
    pred = np.array(patch.evaluate_list(mesh.inverse_map(points, tol=tol).T.tolist())).T
    angular_points[1, :] = pred[-1, :] / center_flux

    return angular_points


if __name__ == "__main__":
    # Discretization
    num_ordinates = 1024

    # Settings
    tol = 1e-8
    eps = 1e-10

    # Create XS server
    xs = {
        "chi": np.array([1.0]),
        "fuel": {
            "nu_fission": np.array([2.84 * 0.081600]),  # 1/cm
            "scatter_gtg": np.array([[[0.225216]]]),  # 1/cm
            "total": np.array([0.32640]),  # 1/cm
        },
    }
    xs_server = Server(xs)

    # Critical radius
    rc = 4.279960  # cm

    # NURBS surfaces
    patch = circle(rc)
    patches = {}
    patches[patch] = "fuel"

    # Create IGA mesh object
    mesh = IGAMesh(patches)

    # Refine mesh
    mesh.refine(0, factor=7, degree=3)

    # Finalize mesh
    mesh.finalize_patches()

    # Set boundary conditions
    mesh.set_boundary_condition(0, 1, 0, "vacuum")
    mesh.set_boundary_condition(0, 1, 1, "vacuum")
    mesh.set_boundary_condition(0, 0, 0, "vacuum")
    mesh.set_boundary_condition(0, 0, 1, "vacuum")

    # Connect patches
    mesh.connect()

    # Plot final mesh
    ax = mesh.plot()
    ax.view_init(90, -90, 0)
    plt.tight_layout()
    plt.savefig("./figs/circle/circle.png", dpi=300)

    # Initialize builder
    builder = OperatorBuilder(
        mesh=mesh,
        xs_server=xs_server,
        num_ordinates=num_ordinates,
        fixed_source=False,
    )

    # Set builder settings
    builder.interp_jacobian = False
    builder.interp_jacobian_det = False
    builder.interp_boundary_jacobian_det = False
    H, S, F = builder.build(use_tt=True)

    # Save TT info
    builder.save_tt_info("./tt_info/circle.csv")

    # Run IGA solver
    k, phi = eig(
        H=H,
        S=S,
        F=F,
        Int_N=builder.Int_N,
        Int_I=builder.Int_I,
        tol=tol,
        eps=eps,
        max_fission_iter=500,
        max_scatter_iter=30,
        sparsity_frac=0,
    )
    phi = phi.flatten()

    # Print eigenvalue error
    print("keff error: {} pcm".format((k - 1) * 1e5))

    # Change control points for phi
    mesh.set_phi(0, phi)
    patch = mesh.patches[0]

    # Plot scalar flux solution at rc
    points = evaluate_boundary(patch)
    plt.clf()
    plt.plot(points[:, 0], points[:, 1])
    plt.hlines(0.2926, -np.pi, np.pi, color="black", linestyles="--")
    plt.ylabel("$\\phi(r = r_c) / \\phi(r = 0)$")
    plt.xlabel("Angle")
    plt.tight_layout()
    plt.savefig("./figs/circle/r_1.png", dpi=300)
    plt.show()

    # Plot scalar flux solution at 0.5rc
    points = evaluate_radius(mesh, patch, 0.5 * rc, tol=1e-10)
    plt.clf()
    plt.plot(points[0, :], points[1, :])
    plt.hlines(0.8093, -np.pi, np.pi, color="black", linestyles="--")
    plt.ylabel("$\\phi(r = 0.5r_c) / \\phi(r = 0)$")
    plt.xlabel("Angle")
    plt.tight_layout()
    plt.savefig("./figs/circle/r_0.5.png", dpi=300)
    plt.show()

    # Plot solution
    plt.clf()
    ax = mesh.plot()
    ax.set_zlabel("$\\phi(x, y)$")
    plt.tight_layout()
    plt.savefig("./figs/circle/phi_1.png", dpi=300)
    plt.show()
