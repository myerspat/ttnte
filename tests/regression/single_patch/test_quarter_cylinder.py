from itertools import product

from power_iteration import power

import pytest
import torch
import numpy as np
from igakit.cad import circle, refine, line, coons

from ttnte import mpi_context
from ttnte.xs.benchmarks import pu239
from ttnte.cad import Patch, BSplineBasis
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
    TTDIGAFirstOrderTransportBackend2D,
)
from ttnte.math import ProductQuadrature
from ttnte.linalg import TTOperator, TTEngine, mm

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]


def evaluate_boundary(patch, rc, dtype):
    # Plot and evaluate boundary flux
    center_flux = patch(torch.tensor([[0, 0]], dtype=dtype))[0][-1]
    points = torch.ones((400, 2))
    points[:200, 0] = torch.linspace(0, 1, 200)
    points[200:, 1] = torch.linspace(0, 1, 200)

    points = patch(points)
    points[:, -1] /= center_flux

    # Convert to angle
    angular_points = torch.zeros((points.shape[0], 2))
    angular_points[:, -1] = points[:, -1]
    angular_points[:, 0] = torch.clamp(
        torch.arcsin(points[(points[:, 0] >= 0), 1] / rc), min=0, max=torch.pi / 4
    )
    return angular_points[angular_points[:, 0].argsort()]


@pytest.mark.parametrize("device, dtype", test_params)
def test_infinite_homogeneous_cylinder(device, dtype):
    """
    This solves the infinite cylinder problem presented in Section 4.1.1
    from the Analytical Benchmark Test Set for Critical Code Verification
    (https://www-sciencedirect-com.proxy.lib.umich.edu/science/article/pii/
    S0149197002000987?fr=RR-2&ref=pdf_download&rr=94263cf04882e830). We
    represent the quarter circle as a NURBS patch with reflective boundary
    conditions, assemble operators in the TT format and solve using the
    alternating minimal energy method (AMEn) in TT format.
    """
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
    rc = 4.279960
    c0 = circle(rc, angle=np.pi / 4)
    c1 = circle(rc, angle=-np.pi / 4).rotate(np.pi / 2)
    l0 = line((0, 0), (rc, 0))
    l1 = line((0, 0), (0, rc))
    c = Patch.from_igakit(
        refine(coons([[l1, c0], [l0, c1]]), [10, 10], 3),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fill,
    )
    assert c.is_finalized()
    assert c.is_rational()
    assert c.ndim == 2
    assert c.ctrlptsw.shape == (13, 13, 3)
    assert c.device == torch.device("cpu")
    assert c.dtype == dtype
    assert c.get_numel(0) == 10
    assert c.get_numel(1) == 10
    assert c.degrees[0] == 3
    assert c.degrees[1] == 3

    mesh = IGAMesh(mpi_context)
    mesh.add_block(c)
    assert not mesh.is_finalized()
    assert not mesh.is_connected()
    assert mesh.num_blocks == 1
    mesh.connect()

    # Set reflective boundary conditions along the y=0 xz-plane and the x=0 yz-plane
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True),
        BoundaryType.REFLECTIVE,
        tol=1e-6,
    )

    # Finalize the mesh, note all boundaries will be vacuum
    mesh.finalize()

    # Create angular quadrature
    qset = ProductQuadrature.gauss_legendre_chebyshev(4, 4, c.ndim)
    qset.to_(torch.device("cpu"), dtype)

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-4 if dtype == torch.float32 else 1e-5
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else 0
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True
    backend = TTDIGAFirstOrderTransportBackend2D(c, qset, xs_server, config)

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
    assert len(Bout) == 4
    assert len(Bin) == 2

    # Check the operators
    for op in [H, S, F] + Bout + Bin:
        if op == None:
            continue

        assert isinstance(op, TTOperator)
        en = op.engine
        assert len(en) == 5
        assert en.device == torch.device("cpu")
        assert en.dtype == dtype
        assert en[0].shape[:-1] == (1, 4, 4)
        assert en[1].shape[1:-1] == (16, 16)
        assert en[2].shape[1:-1] == (
            c.get_numel(0) + c.degrees[0],
            c.get_numel(0) + c.degrees[0],
        )
        assert en[3].shape[1:-1] == (
            c.get_numel(1) + c.degrees[1],
            c.get_numel(1) + c.degrees[1],
        )
        assert en[4].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)

        for i in range(len(en) - 1):
            assert en[i].shape[-1] == en[i + 1].shape[0]

    # Total outflow and inflow boundary operators
    Bout = sum(Bout).round(config.rounding.eps, config.rounding.max_rank).engine
    Bin = sum(Bin).round(config.rounding.eps, config.rounding.max_rank).engine

    # Compute the total operator on the LHS
    A = H.engine - S.engine + Bout - Bin
    A.round_(config.rounding.eps, config.rounding.max_rank)
    F = F.engine

    # Send the TT-operators to GPU if needed
    A.to_(torch.device(device))
    F.to_(torch.device(device))

    k, psi = power(A, F)
    psi.round_(config.rounding.eps, config.rounding.max_rank)
    psi.to_(torch.device("cpu"))
    assert 1e5 * abs(1 - k) < 110

    # Get the angular integral
    angular = TTEngine(
        [w.reshape((1, 1, -1, 1)) for w in qset.get_factored_weights()]
    ).kron(TTEngine.ones(A.n_modes[2:], torch.device("cpu"), dtype).diagonalize())

    # Apply the angular integral to the angular flux
    phi = mm(angular, psi).contract_rank_dim(0).contract_rank_dim(0)
    phi.round_(config.rounding.eps, config.rounding.max_rank)

    # Get the dense version and add this to the patch
    phi = phi.to_dense().squeeze()
    shape = list(c.ctrlptsw.shape)
    shape[-1] += 1
    ctrlptsw = torch.zeros(shape, device=torch.device("cpu"), dtype=dtype)
    ctrlptsw[..., :-2] = c.ctrlptsw[..., :-1]
    ctrlptsw[..., -2] = phi * c.ctrlptsw[..., -1]
    ctrlptsw[..., -1] = c.ctrlptsw[..., -1]

    new_patch = Patch(
        ctrlptsw,
        [BSplineBasis(b.knotvector, b.degree) for b in c.basis],
        is_rational=True,
    )
    new_patch.fill = fill
    new_patch.finalize()

    points = evaluate_boundary(new_patch, rc, dtype)
    torch.testing.assert_close(
        points[:, -1], 0.2926 * torch.ones_like(points[:, -1]), rtol=0.02, atol=0.01
    )

    # Check the relative scalar flux L2 error
    assert (
        np.sqrt(np.trapz((points[:, 1] - 0.2926) ** 2, points[:, 0]))
        / np.sqrt(np.pi / 2 * 0.2926**2)
        < 0.01
    )
