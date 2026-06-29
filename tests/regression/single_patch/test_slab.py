from power_iteration import power

import pytest
import torch
from igakit.cad import refine, line

from ttnte import mpi_context
from ttnte.xs.benchmarks import pu239
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
    DIGAFirstOrderTransportAssembler1D,
)
from ttnte.math import QuadratureSet1D
from ttnte.linalg import Operator, TTEngine, mm

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]


@pytest.mark.parametrize("device, dtype", test_params)
def test_slab(device, dtype):
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
    fills, xs_server = pu239(num_groups=1, device=torch.device("cpu"), dtype=dtype)
    assert fills[0].to_string() == "Pu-239"
    assert xs_server.num_groups == 1

    # Create single-patch geometry (homogeneous circle)
    rc = 2.256751
    c = Patch.from_igakit(
        refine(line((-rc, 0), (rc, 0)), 10, 3),
        device=torch.device("cpu"),
        dtype=dtype,
        fill=fills[0],
    )
    assert c.is_finalized()
    assert not c.is_rational()
    assert c.ndim == 1
    assert c.ctrlptsw.shape == (13, 1)
    assert c.device == torch.device("cpu")
    assert c.dtype == dtype
    assert c.get_numel(0) == 10
    assert c.degrees[0] == 3

    mesh = IGAMesh(mpi_context)
    mesh.add_block(c)
    assert not mesh.is_finalized()
    assert not mesh.is_connected()
    assert mesh.num_blocks == 1
    mesh.connect()

    # Finalize the mesh, note all boundaries will be vacuum
    mesh.finalize()

    # Create 1-D angular quadrature
    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(torch.device("cpu"), dtype)

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6 if dtype == torch.float32 else 1e-5
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else 0
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True
    assembler = DIGAFirstOrderTransportAssembler1D(c, qset, xs_server, config)

    # Assemble individual operators
    assembler.assemble()
    H = assembler.interior_loss_op
    S = assembler.scatter_op
    F = assembler.fission_op
    Bin = assembler.inflow_ops
    Bout = assembler.outflow_ops

    assert all([not b.defined() for b in Bin])
    assert len(Bin) == 2
    assert len(Bout) == 2

    # Check the operators
    for op in [H, S, F] + Bout:
        assert isinstance(op, Operator) and op.is_tt

        en = op.as_tt()
        assert len(en) == 3
        assert en.device == torch.device("cpu")
        assert en.dtype == dtype
        assert en[0].shape[:-1] == (1, 64, 64)
        assert en[1].shape[1:-1] == (
            c.get_numel(0) + c.degrees[0],
            c.get_numel(0) + c.degrees[0],
        )
        assert en[2].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)

        for i in range(len(en) - 1):
            assert en[i].shape[-1] == en[i + 1].shape[0]

    # Total outgoing boundary operator
    Bout = sum(Bout).as_tt().round(config.rounding.eps, config.rounding.max_rank)

    # Compute the total operator on the LHS
    A = (H.as_tt() - S.as_tt() + Bout).round(
        config.rounding.eps, config.rounding.max_rank
    )
    F = F.as_tt()

    # Send the TT-operators to GPU if needed
    A.to_(torch.device(device))
    F.to_(torch.device(device))

    # Run solver
    k, psi = power(A, F)
    psi.round_(config.rounding.eps, config.rounding.max_rank)
    psi.to_(torch.device("cpu"))
    assert 1e5 * abs(1 - k) < 15

    # TODO: Add scalar flux checks once we figure out how to combine the
    # solution with the patch.


# TODO: Add a linearly anisotropic scattering test
