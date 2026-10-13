import multiprocessing

import numpy as np

# NumPy 2.x compatibility for igakit
_numpy_compatibility = {
    "int": int,
    "float": float,
    "complex": complex,
    "intd": np.int_,
    "in1d": np.isin,
    "setmember1d": np.isin,
}

for name, value in _numpy_compatibility.items():
    if name not in np.__dict__:
        setattr(np, name, value)

import torch
from igakit.cad import compat, line, refine, ruled
from igakit.nurbs import NURBS

from ttnte.cad.curves import qtrlobe

from ttnte import mpi_context
from ttnte.visualization.style import get_patch_style
from ttnte.parallel import IGADofHeuristic
from ttnte.xs.benchmarks import c5g7
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
)
from ttnte.math import ProductQuadrature
from ttnte.linalg import Operator, TTEngine, mm
from ttnte.driver import IGATransportDriver2D
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    ExecMode,
    CommMode,
)


def warmup_all_gpus():
    """Forces PyTorch to eagerly initialize the CUDA context, memory allocator, cuBLAS
    (matmul), and cuSOLVER (linalg) on all available GPUs."""
    if not torch.cuda.is_available():
        return

    num_gpus = torch.cuda.device_count()

    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")

        # 1. Warm up the primary CUDA context and memory allocator
        # A simple allocation and pointwise operation
        x = torch.ones(256, 256, device=device)
        y = x * 2.0

        # 2. Warm up cuBLAS (Dense Matrix Multiplication backend)
        _ = torch.matmul(x, y)

        # 3. Warm up cuSOLVER (Linear Algebra backend)
        # This is CRITICAL to prevent the "lazy wrapper should be called at most once"
        # fatal C++ error when background threads do SVD or QR for the first time.
        small_mat = torch.randn(2, 2, device=device)
        _ = torch.linalg.qr(small_mat)
        _ = torch.linalg.svd(small_mat)

        # 4. Synchronize to ensure all asynchronous GPU setup tasks are totally finished
        # before the main Python thread continues.
        torch.cuda.synchronize(device=device)


if __name__ == "__main__":
    # Settings
    dtype = torch.float64
    cpu = torch.device("cpu")

    # Initialize MPI
    mpi_context.init()
    num_threads_per_rank = min(multiprocessing.cpu_count() // mpi_context.world_size, 4)
    torch.set_num_threads(num_threads_per_rank)

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # Get XS information
    fills, xs_server = c5g7(device=cpu, dtype=dtype)

    # Create quarter circle NURBS surface
    def make_curve(points, knots, weights):
        control = np.zeros((len(points), 4), dtype=float)
        control[:, :2] = np.asarray(points) * weights[:, None]
        control[:, 3] = weights
        return NURBS([knots], control)

    def make_line(start, end, knots, weights):
        points = [
            (1.0 - t) * np.asarray(start) + t * np.asarray(end)
            for t in np.linspace(0.0, 1.0, len(weights))
        ]
        return make_curve(points, knots, weights)

    def make_ruled(inner, outer):
        knots = np.asarray(inner.knots[0], dtype=float)
        control = np.stack(
            (
                np.asarray(inner.array, dtype=float),
                np.asarray(outer.array, dtype=float),
            ),
            axis=1,
        )
        return NURBS(
            [knots, np.array([0.0, 0.0, 1.0, 1.0])],
            control,
        )
    
    def make_rational_coons(left, right, bottom, top):
        left, right = compat(left, right)
        bottom, top = compat(bottom, top)

        blend_u = ruled(left, right).transpose()
        blend_v = ruled(bottom, top)

        corners = np.empty((2, 2, 4), dtype=float)
        corners[0, 0] = bottom.array[0, :4]
        corners[1, 0] = bottom.array[-1, :4]
        corners[0, 1] = top.array[0, :4]
        corners[1, 1] = top.array[-1, :4]

        bilinear = NURBS(
            [
                np.array([0.0, 0.0, 1.0, 1.0]),
                np.array([0.0, 0.0, 1.0, 1.0]),
            ],
            corners,
        )

        blend_u, blend_v, bilinear = compat(
            blend_u,
            blend_v,
            bilinear,
        )
        control = (
            blend_u.control
            + blend_v.control
            - bilinear.control
        )

        return NURBS(bilinear.knots, control)
    
    def jacobian_determinant(surface, u, v, step=1e-6):
        u0 = max(0.0, u - step)
        u1 = min(1.0, u + step)
        v0 = max(0.0, v - step)
        v1 = min(1.0, v + step)

        du = (
            np.asarray(surface(u1, v))[:2]
            - np.asarray(surface(u0, v))[:2]
        ) / (u1 - u0)
        dv = (
            np.asarray(surface(u, v1))[:2]
            - np.asarray(surface(u, v0))[:2]
        ) / (v1 - v0)

        return np.linalg.det(np.column_stack((du, dv)))

    # Lightbridge quarter-cell dimensions in centimeters
    D = 1.26
    X = 1.36
    delta = 0.306
    d = 0.04
    dmax = 0.102
    R = 0.297
    a = 0.156

    D2 = 0.5 * D
    y2 = 0.5 * delta
    y1 = y2 - d
    x1 = D2 - R - y2 - dmax
    x2 = x1 + dmax
    half_pitch = 0.5 * X
    displacer_half_width = a / np.sqrt(2.0)

    assert np.isclose(2.0 * (R + x2 + y2), D)
    assert np.isclose((R + d) - R, d)
    assert np.isclose(
        (R + x2 + y2) - (R + d + x1 + y1),
        dmax,
    )

    fuel_outer = qtrlobe(
        outrad=R + d,
        portrs=x1,
        hfwidth=y1,
    )

    guide_outer = qtrlobe(
        outrad=R,
        portrs=x2,
        hfwidth=y2,
    )

    knots = np.asarray(fuel_outer.knots[0], dtype=float)
    weights = np.asarray(fuel_outer.array[:, 3], dtype=float)

    displacer_edge = make_line(
        (0.0, displacer_half_width),
        (displacer_half_width, 0.0),
        knots,
        weights,
    )

    # Split each surrounding material along the diagonal
    displacer_top = displacer_edge.slice(0, 0.0, 0.5)
    displacer_top.remap(0, 0.0, 1.0)

    displacer_right = displacer_edge.slice(0, 0.5, 1.0)
    displacer_right.remap(0, 0.0, 1.0)

    fuel_top = fuel_outer.slice(0, 0.0, 0.5)
    fuel_top.remap(0, 0.0, 1.0)

    fuel_right = fuel_outer.slice(0, 0.5, 1.0)
    fuel_right.remap(0, 0.0, 1.0)

    guide_top = guide_outer.slice(0, 0.0, 0.5)
    guide_top.remap(0, 0.0, 1.0)

    guide_right = guide_outer.slice(0, 0.5, 1.0)
    guide_right.remap(0, 0.0, 1.0)

    # Separate moderator boundaries
    top_outer = make_line(
        (0.0, half_pitch),
        (half_pitch, half_pitch),
        np.asarray(
            guide_top.knots[0],
            dtype=float,
        ),
        np.asarray(
            guide_top.array[:, 3],
            dtype=float,
        ),
    )
    right_outer = make_line(
        (half_pitch, half_pitch),
        (half_pitch, 0.0),
        np.asarray(
            guide_right.knots[0],
            dtype=float,
        ),
        np.asarray(
            guide_right.array[:, 3],
            dtype=float,
        ),
    )

    # Verify the moderator boundaries
    assert np.allclose(
        top_outer(1.0)[:2],
        (half_pitch, half_pitch),
        atol=1e-12,
    )
    assert np.allclose(
        right_outer(0.0)[:2],
        (half_pitch, half_pitch),
        atol=1e-12,
    )

    # Create one central displacer patch
    y_axis = line(
        (0.0, 0.0),
        (0.0, displacer_half_width),
    )
    x_axis = line(
        (0.0, 0.0),
        (displacer_half_width, 0.0),
    )

    central_displacer = make_rational_coons(
        y_axis,
        displacer_right.copy().reverse(0),
        x_axis,
        displacer_top,
    )

    # Seven conforming quarter-cell patches
    displacer = [
        central_displacer,
    ]
    fuel = [
        make_ruled(displacer_top, fuel_top),
        make_ruled(displacer_right, fuel_right),
    ]
    guide = [
        make_ruled(fuel_top, guide_top),
        make_ruled(fuel_right, guide_right),
    ]
    water = [
        make_ruled(guide_top, top_outer),
        make_ruled(guide_right, right_outer),
    ]

    # Check for folded patches
    for surface in displacer + fuel + guide + water:
        determinants = []
        for u in np.linspace(0.01, 0.99, 21):
            for v in np.linspace(0.01, 0.99, 21):
                determinants.append(
                    jacobian_determinant(surface, u, v)
                )

        assert min(determinants) > 1e-8

    # Create the mesh
    mesh = IGAMesh(mpi_context)

    fill_groups = [
        fills[-1],
        fills[0],
        fills[0],
        fills[-3],
        fills[-3],
        fills[-2],
        fills[-2],
    ]

    for fill, surface in zip(
        fill_groups,
        displacer + fuel + guide + water,
    ):
        mesh.add_block(
            Patch.from_igakit(
                refine(surface, 10, 2),
                device=cpu,
                dtype=dtype,
                fill=fill,
            )
        )

    # Connect patches
    mesh.connect()

    # Set the boundary conditions
    mesh.set_axis_aligned_conditions(
        BCPlane(
            x_min=True,
            y_min=True,
            x_max=True,
            y_max=True,
        ),
        BoundaryType.REFLECTIVE,
        tol=1e-6,
    )
    mesh.finalize()

    counts = {
        BoundaryType.INTERNAL: 0,
        BoundaryType.REFLECTIVE: 0,
        BoundaryType.VACUUM: 0,
        BoundaryType.DEGENERATE: 0,
        BoundaryType.UNKNOWN: 0,
    }

    for patch in mesh.blocks:
        for dim in range(2):
            for is_upper in (False, True):
                boundary = patch.get_boundary_info(
                    dim,
                    is_upper,
                )
                counts[boundary.type] += 1

                if boundary.type == BoundaryType.INTERNAL:
                    assert len(boundary.connections) == 1

    assert mesh.num_blocks == 7
    assert counts[BoundaryType.INTERNAL] == 18
    assert counts[BoundaryType.REFLECTIVE] == 10
    assert counts[BoundaryType.VACUUM] == 0
    assert counts[BoundaryType.DEGENERATE] == 0
    assert counts[BoundaryType.UNKNOWN] == 0

    from pathlib import Path

    backend = "matplotlib"
    style = get_patch_style(backend)
    style.mesh.cmap = {
        fills[-1].to_string(): "purple",
        fills[0].to_string(): "orange",
        fills[-3].to_string(): "gray",
        fills[-2].to_string(): "cornflowerblue",
    }

    figure_path = (
        Path(__file__).resolve().parent
        / "figs"
        / "cruciform_pincell_2d.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    mesh.plot(
        resolution=37,
        show_ctrlpts=False,
        show_ctrlnet=False,
        show_boundary=True,
        backend=backend,
        filename=str(figure_path),
        style=style,
    )

    # Create angular quadrature
    qset = ProductQuadrature.gauss_legendre_chebyshev(16, 16, 2)
    qset.to_(torch.device("cpu"), dtype)

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = False

    # Create the transport driver
    driver = IGATransportDriver2D(mesh, xs_server, mpi_context)

    # Distribute patches among MPI ranks
    driver.distribute([IGADofHeuristic()])

    # Run the assembler
    driver.assemble(qset, config)
    print("Seven-patch transport assembly completed successfully.")
    raise SystemExit(0) # Comment out this line to run the solver

    # Warmup GPUs
    warmup_all_gpus()

    # Create Block-Jacobi DD strategy
    config = DDSolverConfig(
        tol=5e-7,
        max_iter=100,
        eps=5e-8,
        use_gpu=True,
        memory_policy=MemoryPolicy.RESIDENT,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=True,
    )
    config.inner_forcing = 0.1
    config.eps_forcing = 0.01
    strategy = BlockJacobiStrategy(config)
    strategy.set_local_solver(
        AMEnSolver(
            nswp=10,
            eps=5e-8,
            kickrank=6,
            local_iterations=200,
            resets=4,
            rmax=500,
        )
    )

    k = driver.solve_eigenvalue(strategy, tol=1e-6, max_iter=500, verbose=True)
