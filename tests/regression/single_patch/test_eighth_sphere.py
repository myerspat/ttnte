from itertools import product

from power_iteration import power

import pytest
import torch
import numpy as np
from igakit.cad import refine, NURBS

from ttnte import mpi_context
from ttnte.xs.benchmarks import pu239
from ttnte.cad.surfaces import circle
from ttnte.cad import Patch, BSplineBasis
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
    TTDIGAFirstOrderTransportBackend3D,
)
from ttnte.math import ProductQuadrature
from ttnte.linalg import TTOperator, TTEngine, mm

test_params = [
    # ("cpu", torch.float32),
    # ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]


def create_solid_octant_sphere(radius=1.0):
    """Generates a 1/8th solid sphere (octant) with axis-aligned reflective boundaries.

    U: Radial direction (Degree 1, bounds the origin to the outer shell)
    V: Azimuthal direction (Degree 2, sweeps from X-axis to Y-axis)
    W: Elevation direction (Degree 2, sweeps from XY-plane to Z-axis)
    """
    s2 = np.sqrt(2.0)

    # Dimensions: [U (radial), V (azimuth), W (elevation), 4 (homogeneous coords)]
    # U = 2 control points, V = 3 control points, W = 3 control points
    ctrl = np.zeros((2, 3, 3, 4))

    # --- Outer Shell (U = 1) ---
    # Homogeneous coordinates are computed as [x*w, y*w, z*w, w]

    # W = 0 (Equator / XY plane)
    ctrl[1, 0, 0] = [radius, 0, 0, 1.0]
    ctrl[1, 1, 0] = [radius * s2 / 2, radius * s2 / 2, 0, s2 / 2]
    ctrl[1, 2, 0] = [0, radius, 0, 1.0]

    # W = 1 (Mid elevation)
    ctrl[1, 0, 1] = [radius * s2 / 2, 0, radius * s2 / 2, s2 / 2]
    ctrl[1, 1, 1] = [radius * 0.5, radius * 0.5, radius * 0.5, 0.5]
    ctrl[1, 2, 1] = [0, radius * s2 / 2, radius * s2 / 2, s2 / 2]

    # W = 2 (North pole / Z-axis)
    ctrl[1, 0, 2] = [0, 0, radius, 1.0]
    ctrl[1, 1, 2] = [0, 0, radius * s2 / 2, s2 / 2]
    ctrl[1, 2, 2] = [0, 0, radius, 1.0]

    # --- Inner Core (U = 0) ---
    # All x,y,z coordinates collapse to exactly 0 (the origin).
    # CRITICAL: The weights (index 3) must perfectly match the Outer Shell
    # to guarantee straight radial elements and prevent Jacobian inversion.
    for v in range(3):
        for w in range(3):
            weight = ctrl[1, v, w, 3]
            ctrl[0, v, w] = [0.0, 0.0, 0.0, weight]

    # --- Knot Vectors ---
    # U is degree 1 (linear radial interpolation)
    knots_u = [0, 0, 1, 1]

    # V and W are degree 2 (quadratic circular arcs)
    knots_v = [0, 0, 0, 1, 1, 1]
    knots_w = [0, 0, 0, 1, 1, 1]

    # Generate and return the single-patch NURBS volume
    return NURBS([knots_u, knots_v, knots_w], ctrl)


@pytest.mark.parametrize("device, dtype", test_params)
def test_homogeneous_sphere(device, dtype):
    """"""
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Initialize the context
    mpi_context.init()

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # Get XS info
    fill, xs_server = pu239(num_groups=1, device=torch.device("cpu"), dtype=dtype)
    assert fill.to_string() == "Pu-239"
    assert xs_server.num_groups == 1

    # Create single-patch geometry (homogeneous circle)
    rc = 6.082547
    c = Patch.from_igakit(
        refine(create_solid_octant_sphere(rc), 4, 2),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fill,
    )

    mesh = IGAMesh(mpi_context)
    mesh.add_block(c)
    mesh.connect()

    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True, z_min=True),
        BoundaryType.REFLECTIVE,
        tol=1e-6,
    )

    mesh.finalize()

    qset = ProductQuadrature.gauss_legendre_chebyshev(4, 4, c.ndim)
    qset.to_(torch.device("cpu"), dtype)

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-4 if dtype == torch.float32 else 1e-5
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = False
    backend = TTDIGAFirstOrderTransportBackend3D(c, qset, xs_server, config)

    # Assemble individual operators
    H = backend.assemble_loss_operator()
    S = backend.assemble_scatter_operator()
    F = backend.assemble_fission_operator()

    Bout = []
    Bin = []
    for dim, is_upper in product(range(c.ndim), [False, True]):
        B = backend.assemble_boundary_operators(dim, is_upper)
        if B[0] != None:
            Bout.append(B[0])
        if B[1] != None:
            Bin.append(B[1])
    # assert len(Bout) == 4
    # assert len(Bin) == 2

    # Check the operators
    for op in [H, S, F] + Bout + Bin:
        if op == None:
            continue

        # assert isinstance(op, TTOperator)
        # en = op.engine
        # assert len(en) == 5
        # assert en.device == torch.device("cpu")
        # assert en.dtype == dtype
        # assert en[0].shape[:-1] == (1, 4, 4)
        # assert en[1].shape[1:-1] == (16, 16)
        # assert en[2].shape[1:-1] == (
        #     c.get_numel(0) + c.degrees[0],
        #     c.get_numel(0) + c.degrees[0],
        # )
        # assert en[3].shape[1:-1] == (
        #     c.get_numel(1) + c.degrees[1],
        #     c.get_numel(1) + c.degrees[1],
        # )
        # assert en[4].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)
        #
        # for i in range(len(en) - 1):
        #     assert en[i].shape[-1] == en[i + 1].shape[0]

    # Total outflow and inflow boundary operators
    Bout = sum(Bout).round(config.rounding.eps, config.rounding.max_rank).engine
    Bin = sum(Bin).round(config.rounding.eps, config.rounding.max_rank).engine

    H = (
        H.engine.contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
    )
    S = (
        S.engine.contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
    )
    F = (
        F.engine.contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
    )
    Bout = (
        Bout.contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
    )
    Bin = (
        Bin.contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
        .contract_rank_dim_(2)
    )

    H.round_(config.rounding.eps, config.rounding.max_rank)
    S.round_(config.rounding.eps, config.rounding.max_rank)
    F.round_(config.rounding.eps, config.rounding.max_rank)
    Bout.round_(config.rounding.eps, config.rounding.max_rank)
    Bin.round_(config.rounding.eps, config.rounding.max_rank)

    print("H")
    for core in H:
        print(core.shape)
    print("S")
    for core in S:
        print(core.shape)
    print("F")
    for core in S:
        print(core.shape)
    print("Bout")
    for core in Bout:
        print(core.shape)
    print("Bin")
    for core in Bin:
        print(core.shape)

    # Compute the total operator on the LHS
    # A = H.engine - S.engine + Bout - Bin
    A = H - S + Bout - Bin
    A.round_(config.rounding.eps, config.rounding.max_rank)

    # Send the TT-operators to GPU if needed
    A.to_(torch.device(device, 2))
    F.to_(torch.device(device, 2))

    k, psi = power(A, F)
    psi.round_(config.rounding.eps, config.rounding.max_rank)
    psi.to_(torch.device("cpu"))
