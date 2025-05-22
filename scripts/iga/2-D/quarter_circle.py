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

    # Create quarter circle NURBS surface
    c0 = cad.circle(radius=rc, angle=np.pi / 2)
    l0 = cad.line(p0=(0, 0), p1=(0, 0))
    patch = cad.ruled(l0, c0)

    # NURBS surfaces
    patches = {}
    patches[patch] = "fuel"

    # Create IGA mesh object
    mesh = IGAMesh(patches)

    # Refine mesh
    mesh.refine(0, factor=[3, 9], degree=3)

    # Finalize mesh
    mesh.finalize_patches()

    # Set boundary conditions
    mesh.set_boundary_condition(0, 1, 0, "vacuum")
    mesh.set_boundary_condition(0, 1, 1, "vacuum")
    mesh.set_boundary_condition(0, 0, 0, "reflective")
    mesh.set_boundary_condition(0, 0, 1, "reflective")

    # Connect patches
    mesh.connect()

    # Plot final mesh
    ax = mesh.plot()
    ax.view_init(90, -90, 0)
    plt.tight_layout()
    plt.savefig("./figs/quarter_circle/quarter_circle.png", dpi=300)

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
    builder.save_tt_info("./tt_info/quarter_circle.csv")

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

    # Calculate eigenvalue error
    print("keff error: {} pcm".format((k - 1) * 1e5))

    # Change control points for phi
    mesh.set_phi(0, phi)
    patch = mesh.patches[0]

    # Plot scalar flux solution at rc
    points = evaluate_boundary(patch)
    plt.clf()
    plt.plot(points[:, 0], points[:, 1])
    plt.hlines(0.2926, 0, np.pi / 2, color="black", linestyles="--")
    plt.legend(["IGA", "Benchmark"])
    plt.ylabel("$\\phi(r = r_c) / \\phi(r = 0)$")
    plt.xlabel("Angle")
    plt.tight_layout()
    plt.savefig("./figs/quarter_circle/r_1.png", dpi=300)
    plt.show()

    # Plot scalar flux solution at 0.5rc
    points = evaluate_radius(mesh, patch, radius=0.5 * rc)
    plt.clf()
    plt.plot(points[0, :], points[1, :])
    plt.hlines(0.8093, 0, np.pi / 2, color="black", linestyles="--")
    plt.ylabel("$\\phi(r = 0.5r_c) / \\phi(r = 0)$")
    plt.xlabel("Angle")
    plt.legend(["IGA", "Benchmark"])
    plt.tight_layout()
    plt.savefig("./figs/quarter_circle/r_0.5.png", dpi=300)
    plt.show()

    # Plot solution
    plt.clf()
    ax = mesh.plot()
    ax.set_zlabel("$\\phi(x, y)$")
    plt.tight_layout()
    plt.savefig("./figs/quarter_circle/phi_1.png", dpi=300)
    plt.show()
