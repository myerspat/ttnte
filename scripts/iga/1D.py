import itertools
import time

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from geometry import Geometry as Geometry_IGA
from quimb.tensor import MatrixProductOperator
from reed import reed
from solve import eig, fixed_source
from tt import build_operators

from tt_nte.benchmarks import research_reactor_multi_region
from tt_nte.benchmarks.crit_verif import _create_1d_mesh
from tt_nte.geometry import Geometry as Geometry_FD
from tt_nte.methods import DiscreteOrdinates
from tt_nte.xs import Server


def solve_fixed_iga(
    xs_server,
    geometry,
    num_ordinates,
    num_points,
    tol=1e-6,
    eps=1e-10,
    max_iter=500,
    single_precision=False,
    sparsity_frac=0.2,
):
    # Build TT operators
    H, S, q, Int_N = build_operators(
        geometry,
        xs_server,
        num_ordinates=num_ordinates,
        num_points=num_points,
        fixed_source=True,
    )

    return fixed_source(
        H,
        S,
        q,
        Int_N,
        tol=tol,
        eps=eps,
        max_iter=max_iter,
        single_precision=single_precision,
        sparsity_frac=sparsity_frac,
    )


def solve_eig_iga(
    xs_server,
    geometry,
    num_ordinates,
    num_points,
    tol=1e-6,
    eps=1e-10,
    max_fission_iter=500,
    max_scatter_iter=20,
    single_precision=False,
    sparsity_frac=0.2,
):
    # Build TT operators
    H, S, F, Int_N = build_operators(
        geometry, xs_server, num_ordinates=num_ordinates, num_points=num_points
    )

    # Define integral for total fission source
    Int_I = MatrixProductOperator(
        [
            np.block(
                [np.ones(int(num_ordinates / 2)), np.zeros(int(num_ordinates / 2))]
            ).reshape((1, -1, 1)),
            np.ones(xs_server.num_groups).reshape((1, 1, -1, 1)),
            np.ones(geometry.num_patches).reshape((1, 1, -1, 1)),
            np.array(
                [0] + (geometry.patches[0].num_dofs - 1) * [1], dtype=float
            ).reshape((1, 1, -1)),
        ],
        shape="ludr",
    ) + MatrixProductOperator(
        [
            np.block(
                [np.zeros(int(num_ordinates / 2)), np.ones(int(num_ordinates / 2))]
            ).reshape((1, -1, 1)),
            np.ones(xs_server.num_groups).reshape((1, 1, -1, 1)),
            np.ones(geometry.num_patches).reshape((1, 1, -1, 1)),
            np.array(
                (geometry.patches[0].num_dofs - 1) * [1] + [0], dtype=float
            ).reshape((1, 1, -1)),
        ],
        shape="ludr",
    )

    # Run eigenvalue solver
    return eig(
        H=H,
        S=S,
        F=F,
        Int_N=Int_N,
        Int_I=Int_I,
        tol=tol,
        eps=eps,
        max_fission_iter=max_fission_iter,
        max_scatter_iter=max_scatter_iter,
        single_precision=single_precision,
        sparsity_frac=sparsity_frac,
    )


def solve_eig_fd(
    xs_server,
    geometry,
    num_ordinates,
    tol=1e-6,
    eps=1e-10,
    max_fission_iter=500,
    max_scatter_iter=20,
    single_precision=False,
    sparsity_frac=0.2,
):
    # Create SN discretization
    SN = DiscreteOrdinates(
        xs_server=xs_server,
        geometry=geometry,
        num_ordinates=num_ordinates,
    )

    # Get operators
    H = SN.H.to_quimb()
    S = SN.S.to_quimb()
    F = SN.F.to_quimb()

    # Define integral for total fission source
    Int_I = MatrixProductOperator(
        [
            np.block(
                [np.ones(int(num_ordinates / 2)), np.zeros(int(num_ordinates / 2))]
            ).reshape((1, -1, 1)),
            np.ones(xs_server.num_groups).reshape((1, 1, -1, 1)),
            np.array([0] + geometry.dx.flatten().tolist(), dtype=float).reshape(
                (1, 1, -1)
            ),
        ],
        shape="ludr",
    ) + MatrixProductOperator(
        [
            np.block(
                [np.zeros(int(num_ordinates / 2)), np.ones(int(num_ordinates / 2))]
            ).reshape((1, -1, 1)),
            np.ones(xs_server.num_groups).reshape((1, 1, -1, 1)),
            np.array(geometry.dx.flatten().tolist() + [0], dtype=float).reshape(
                (1, 1, -1)
            ),
        ],
        shape="ludr",
    )

    # Run eigenvalue solver
    k, phi = eig(
        H=H,
        S=S,
        F=F,
        Int_N=SN.Int_N.to_quimb(),
        Int_I=Int_I,
        tol=tol,
        eps=eps,
        max_fission_iter=max_fission_iter,
        max_scatter_iter=max_scatter_iter,
        single_precision=single_precision,
        sparsity_frac=sparsity_frac,
    )
    return k, phi / np.linalg.norm(phi.flatten(), 2)


def compute_scalar_flux(geometry, ctrlpts, x):
    def compute_phi(patch_idx, x):
        # Get current patch
        patch = geometry.patches[patch_idx]

        # Map x to knot
        x -= sum([p.L for p in geometry.patches[:patch_idx]])
        knot = x / patch.L

        # Find span of knot and compute non-vanishing basis functions
        span = patch.find_spans([knot])[0]
        bf = patch.basis_functions([knot])[0, :]

        # Calculate flux
        return ctrlpts[:, patch_idx, span - patch.degree : span + 1] @ bf

    # Sort x to each patch
    patch_idxs = (
        np.searchsorted(
            np.cumsum([0] + [p.L for p in geometry_iga.patches]), x, "right"
        )
        - 1
    )
    patch_idxs[x >= geometry_iga.L] -= 1

    # Calculate scalar flux
    phi = np.array(list(map(compute_phi, patch_idxs, x))).T
    return ctrlpts, phi


def plot_scalar_flux(
    x, phi_fd, phi_iga, ctrlpts, file_path, label1="FD", label2="IGA", show_ctrlpts=True
):
    num_groups = phi_fd.shape[0]
    ctrlpts = ctrlpts.reshape((ctrlpts.shape[0], -1))

    plt.clf()
    for g in range(num_groups):
        if show_ctrlpts:
            plt.plot(
                [
                    c[0]
                    for c in list(
                        itertools.chain.from_iterable(
                            [patch.ctrlpts for patch in geometry_iga.patches]
                        )
                    )
                ],
                ctrlpts[g, :],
                "--ok",
                label="Control polygon" if g == 0 else None,
            )
        plt.plot(
            x,
            phi_fd[g, :],
            label="$\\phi^{" + label1 + "}_" + str(g + 1) + "(x)$",
        )
        plt.plot(
            x,
            phi_iga[g, :],
            "--",
            label="$\\phi^{" + label2 + "}_" + str(g + 1) + "(x)$",
        )

    plt.legend()
    plt.xlabel("$x~(cm)$")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)


if __name__ == "__main__":
    # Discretization
    num_nodes = 1024
    num_ordinates = 64

    # Settings
    tol = 1e-6  # Fission source convergence
    eps = 1e-10  # TT compression threshold

    # =================================================================
    # 2-group Pu Brick (1-Patch System)
    # =================================================================

    # Cross section data
    xs = {
        "chi": np.array([0.575, 0.425]),
        "fuel": {
            "total": np.array([0.2208, 0.3360]),
            "nu_fission": np.array([0.29016, 0.2503392]),
            "scatter_gtg": np.array([[[0.0792, 0.0], [0.0432, 0.23616]]]),
        },
    }
    xs_server = Server(xs)

    # Define finite difference discretized geometry
    gmsh.initialize()
    gmsh.model.add("Pu Brick")
    gmsh.option.setNumber("General.Terminal", 0)
    _create_1d_mesh(
        gmsh.model,
        np.array(["fuel"]),
        np.array([3.591204]),
        np.array([num_nodes]),
        "vacuum",
        "vacuum",
    )
    geometry_fd = Geometry_FD(gmsh.model)
    gmsh.finalize()

    # Define NURBS discretized geometry
    geometry_iga = Geometry_IGA([{"mat": "fuel", "xl": 0, "xr": 3.591204}])

    # Refine mesh and basis
    geometry_iga.knot_insert()
    geometry_iga.knot_insert()
    geometry_iga.degree_elevate()

    # Run finite difference version of TT
    k_fd, phi_fd = solve_eig_fd(
        xs_server,
        geometry_fd,
        num_ordinates=num_ordinates,
        tol=tol,
        eps=eps,
    )

    # Run IGA solver
    k_iga, ctrlpts = solve_eig_iga(
        xs_server=xs_server,
        geometry=geometry_iga,
        num_ordinates=num_ordinates,
        num_points=geometry_iga.degree + 1,
        tol=tol,
        eps=eps,
    )

    # Calculate scalar flux at FD node locations
    x = np.cumsum([0] + geometry_fd.dx.flatten().tolist())
    ctrlpts, phi_iga = compute_scalar_flux(geometry_iga, ctrlpts, x)
    ctrlpts, phi_iga = ctrlpts / np.linalg.norm(
        phi_iga.flatten(), 2
    ), phi_iga / np.linalg.norm(phi_iga.flatten(), 2)
    print("Phi Control Points:", ctrlpts)

    # Calculate scalar flux and eigenvalue error
    print(
        "Scalar Flux L2 Error:",
        np.linalg.norm(phi_iga - phi_fd, 2) / np.linalg.norm(phi_fd, 2),
    )
    print("Eigenvalue Relative Error:", np.abs(k_iga - k_fd) / np.abs(k_fd))

    # Plot scalar fluxes
    plot_scalar_flux(x, phi_fd, phi_iga, ctrlpts, file_path="./figs/homogeneous.png")

    # =================================================================
    # 2-group Research Reactor (3-Patch System)
    # =================================================================

    # Load XS and FD geometry data
    # xs_server, geometry_fd = research_reactor_multi_region([265, 1521, 264], "vacuum")
    xs_server, geometry_fd = research_reactor_multi_region([265, 760], "reflective")

    # Create IGA geometry
    geometry_iga = Geometry_IGA(
        [
            {"mat": "moderator", "xl": 0, "xr": 1.126151},
            {"mat": "fuel", "xl": 1.126151, "xr": 6.696802 + 1.126151},
        ],
        left_bc="vacuum",
        right_bc="reflective",
    )
    # geometry_iga = Geometry_IGA(
    #     [
    #         {"mat": "moderator", "xl": 0, "xr": 1.126151},
    #         {"mat": "fuel", "xl": 1.126151, "xr": 2 * 6.696802 + 1.126151},
    #         {
    #             "mat": "moderator",
    #             "xl": 2 * 6.696802 + 1.126151,
    #             "xr": 2 * (6.696802 + 1.126151),
    #         },
    #     ],
    # )

    # Refine mesh and basis
    geometry_iga.knot_insert()
    geometry_iga.knot_insert()
    geometry_iga.degree_elevate()

    # Run finite difference version of TT
    k_fd, phi_fd = solve_eig_fd(
        xs_server,
        geometry_fd,
        num_ordinates=num_ordinates,
        tol=tol,
        eps=eps,
    )

    # Run IGA solver
    k_iga, ctrlpts = solve_eig_iga(
        xs_server=xs_server,
        geometry=geometry_iga,
        num_ordinates=num_ordinates,
        num_points=geometry_iga.degree + 1,
        tol=tol,
        eps=eps,
    )

    # Calculate scalar flux at FD node locations
    x = np.cumsum([0] + geometry_fd.dx.flatten().tolist())
    ctrlpts, phi_iga = compute_scalar_flux(geometry_iga, ctrlpts, x)
    ctrlpts, phi_iga = ctrlpts / np.linalg.norm(
        phi_iga.flatten(), 2
    ), phi_iga / np.linalg.norm(phi_iga.flatten(), 2)
    print("Phi Control Points:", ctrlpts)

    # Calculate scalar flux and eigenvalue error
    print(
        "Scalar Flux L2 Error:",
        np.linalg.norm(phi_iga - phi_fd, 2) / np.linalg.norm(phi_fd, 2),
    )

    print("Eigenvalue Relative Error:", np.abs(k_iga - k_fd) / np.abs(k_fd))

    # Plot scalar fluxes
    plot_scalar_flux(x, phi_fd, phi_iga, ctrlpts, file_path="./figs/two_patch.png")

    # =================================================================
    # Reed Cell Example
    # =================================================================

    # Define XSs
    xs = {
        "chi": np.array([0]),
        "fuel": {
            "total": np.array([50.0]),
            "scatter_gtg": np.array([[[0.0]]]),
            "nu_fission": np.array([0.0]),
        },
        "can": {
            "total": np.array([5.0]),
            "scatter_gtg": np.array([[[0.0]]]),
            "nu_fission": np.array([0.0]),
        },
        "void": {
            "total": np.array([0.0]),
            "scatter_gtg": np.array([[[0.0]]]),
            "nu_fission": np.array([0.0]),
        },
        "moderator": {
            "total": np.array([1.0]),
            "scatter_gtg": np.array([[[0.9]]]),
            "nu_fission": np.array([0.0]),
        },
    }
    xs_server = Server(xs)

    # Meshgrid of knots and degree
    degrees = [0, 1, 2]
    knot_additions, degrees = np.meshgrid(np.arange(9) + 1, degrees)
    average_h = np.zeros(knot_additions.shape)
    times = np.zeros(knot_additions.shape)
    errors = np.zeros(knot_additions.shape)

    for i in range(knot_additions.shape[0]):
        for j in range(knot_additions.shape[1]):
            # Get IGA geometry
            geometry_iga = Geometry_IGA(
                [
                    {"mat": "fuel", "xl": 0, "xr": 2, "source": 50.0},
                    {"mat": "can", "xl": 2, "xr": 3},
                    {"mat": "void", "xl": 3, "xr": 5},
                    {"mat": "moderator", "xl": 5, "xr": 6, "source": 1.0},
                    {"mat": "moderator", "xl": 6, "xr": 8},
                ],
                left_bc="reflective",
                right_bc="vacuum",
            )

            # Refine mesh
            geometry_iga.knot_insert(int(knot_additions[i, j]))
            if degrees[i, j] > 0:
                geometry_iga.degree_elevate(int(degrees[i, j]))

            assert geometry_iga.degree == degrees[i, j] + 1
            assert geometry_iga.patches[0].degree == degrees[i, j] + 1

            # Run IGA solver
            start = time.time()
            ctrlpts = solve_fixed_iga(
                xs_server,
                geometry_iga,
                num_ordinates=8,
                num_points=geometry_iga.degree + 1,
                max_iter=1000,
            )
            times[i, j] = time.time() - start

            # Calculate average mesh size
            x = np.unique(
                [
                    c[0]
                    for c in list(
                        itertools.chain.from_iterable(
                            [patch.ctrlpts for patch in geometry_iga.patches]
                        )
                    )
                ]
            )
            average_h[i, j] = np.average(x[1:] - x[:-1])

            # Get exact solution
            x = np.linspace(0, 8, 4000)
            phi_exact = reed(x).reshape((1, -1))
            ctrlpts, phi_iga = compute_scalar_flux(geometry_iga, ctrlpts, x)
            print("Phi Control Points:", ctrlpts)

            errors[i, j] = np.sqrt(np.trapz((phi_exact - phi_iga) ** 2, x))
            print(
                "Scalar Flux L2 Error:",
                errors[i, j],
            )

            # Plot scalar fluxes
            plot_scalar_flux(
                x,
                phi_exact,
                phi_iga,
                ctrlpts,
                f"./figs/reed/d{degrees[i, j] + 1}_{np.round(average_h[i, j], 4)}.png",
                "Exact",
                show_ctrlpts=False,
            )

    # Plot error
    plt.clf()
    shape = ["o", "v", "s", "*"]
    for i in range(degrees.shape[0]):
        plt.plot(average_h[i, :], errors[i, :], "-" + shape[i], label=f"PDG-{i + 1}")

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Average $h$ (cm)")
    plt.ylabel("Scalar Flux L2 Error")
    plt.tight_layout()
    plt.savefig("./figs/reed/convergence.png", dpi=300)

    plt.clf()
    for i in range(degrees.shape[0]):
        plt.plot(average_h[i, :], times[i, :], "-" + shape[i], label=f"PDG-{i + 1}")

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Average $h$ (cm)")
    plt.ylabel("Execution Time (s)")
    plt.tight_layout()
    plt.savefig("./figs/reed/times.png", dpi=300)

    # Approximate order of convergence
    print("Order of Convergence")
    print(
        np.log10(errors[:, -1] / errors[:, -2])
        / np.log10(average_h[:, -1] / average_h[:, -2])
    )
