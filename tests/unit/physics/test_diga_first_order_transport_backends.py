import pytest
import torch
import gc
import time
from igakit.cad import refine, grid, extrude, line

from ttnte.cad.curves import ellipse
from ttnte.cad.surfaces import circle
from ttnte.cad import Patch, BSplineBasis
from ttnte.physics import (
    DGTransportAssemblerConfig,
    DenseDIGAFirstOrderTransportBackend1D,
    DenseDIGAFirstOrderTransportBackend2D,
    DenseDIGAFirstOrderTransportBackend3D,
    TTDIGAFirstOrderTransportBackend1D,
    TTDIGAFirstOrderTransportBackend2D,
    TTDIGAFirstOrderTransportBackend3D,
)
from ttnte.xs import Server, Material
from ttnte.math import ProductQuadrature, QuadratureSet1D
from ttnte.linalg import TTEngine, mm

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    # ("cuda", torch.float32),
    # ("cuda", torch.float64),
]


def filler_xs_and_quadrature(device, dtype, ndim=3):
    mat = Material("mat")
    mat.total = 5 * torch.ones((3), device=device, dtype=dtype)
    mat.scatter_gtg = torch.ones((3, 3, 3), device=device, dtype=dtype)
    mat.finalize()

    server = Server()
    server.add_material(mat)
    server.finalize()

    if ndim > 1:
        qset = ProductQuadrature.gauss_legendre_chebyshev(4, 4, ndim)
    else:
        qset = QuadratureSet1D.gauss_legendre(8)
    qset.to_(torch.device(device), dtype)
    return mat.label, server, qset


@pytest.mark.parametrize("device, dtype", test_params)
def test_1d_line(device, dtype):
    # Tolerances for this test
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    dtol = 1e-2 if dtype == torch.float32 else 1e-8

    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get test XSs and quadrature set
    mat_label, xs_server, qset = filler_xs_and_quadrature(device, dtype, 1)

    # Get the exact curve from igakit
    c = Patch.from_igakit(
        line((0, 0), (0, 1)), device=device, dtype=dtype, fill=mat_label
    )
    assert not c.is_rational()

    # Create NURBS version
    shape = list(c.ctrlptsw.shape)
    shape[-1] += 1
    ctrlptsw = torch.ones(shape, device=device, dtype=dtype)
    ctrlptsw[..., :-1] = c.ctrlptsw
    c_test = Patch(
        ctrlptsw,
        [BSplineBasis(b.knotvector, b.degree) for b in c.basis],
        is_rational=True,
    )
    c_test.fill = mat_label
    c_test.finalize()
    assert c_test.is_rational()

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6 if dtype == torch.float32 else 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else 0
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True
    dense_assembler = DenseDIGAFirstOrderTransportBackend1D(
        c_test, qset, xs_server, config
    )
    tt_assembler = TTDIGAFirstOrderTransportBackend1D(c_test, qset, xs_server, config)

    # =================================================================
    # Test assemble_basis() method
    expected = c.evaluate_all_basis(dense_assembler.quad_points, 0)

    actual = dense_assembler.assemble_basis()
    assert actual.shape == (
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
    )
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    actual = tt_assembler.assemble_basis()
    assert actual[0].shape == (
        1,
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
        1,
    )
    torch.testing.assert_close(actual.to_dense(), expected, atol=tol, rtol=tol)

    # =================================================================
    # Test assemble_basis_ders() method
    expected = c.evaluate_all_basis(dense_assembler.quad_points, 1, False)

    actual = dense_assembler.assemble_basis_ders()
    assert actual.shape[1] == 2
    assert actual[:, 0, :].shape == (
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
    )
    assert actual[:, 1, :].shape == (
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
    )
    torch.testing.assert_close(actual[:, 0, :], expected[:, 0, :], atol=tol, rtol=tol)
    torch.testing.assert_close(actual[:, 1, :], expected[:, 1, :], atol=tol, rtol=tol)

    actual = tt_assembler.assemble_basis_ders()
    assert actual[0][0].shape == (
        1,
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
        1,
    )
    assert actual[1][0].shape == (
        1,
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
        1,
    )
    torch.testing.assert_close(
        actual[0].to_dense(), expected[:, 0, :], atol=tol, rtol=tol
    )
    torch.testing.assert_close(
        actual[1].to_dense(), expected[:, 1, :], atol=dtol, rtol=dtol
    )

    # =================================================================
    # Test assemble_scattering_kernel() method

    actual = tt_assembler.assemble_scattering_kernel()
    assert len(actual) == 2
    assert actual[0].shape[:-1] == (1, 8, 8)
    assert actual[1].shape[1:] == (3, 3, 1)
    assert actual[0].shape[-1] == actual[1].shape[0]

    # Apply angular weights to make this operator an integral
    actual = tt_assembler.apply_angular_weights(actual, [0])

    # Test isotropic flux case
    phi_iso = TTEngine.ones(actual.n_modes, actual.device, actual.dtype)

    # Check result is the 0th order group-to-group scattering XS tensor
    torch.testing.assert_close(
        mm(actual, phi_iso).to_dense().squeeze(),
        xs_server.get_material(mat_label)
        .scatter_gtg[0, :, :]
        .sum(0)
        .reshape((1, -1))
        .expand((8, 3)),
    )

    # =================================================================
    # Test assemble_jacobian() and assemble_jacobian_inverse()

    jac = tt_assembler.assemble_jacobian()
    jac_inv = tt_assembler.assemble_jacobian_inverse()

    actual = jac[0][0] * jac_inv[0][0]

    # Check we only get ones (identity)
    torch.testing.assert_close(
        actual[0], torch.ones_like(actual[0]), atol=tol, rtol=tol
    )

    # =================================================================
    # Test assemble_integral_mapping() method

    # Check the integral evaluates to the length of the line
    actual = tt_assembler.assemble_integral_mapping()
    assert len(actual) == 1
    assert actual[0].shape == (1, c.get_numel(0) * (c.degrees[0] + 1), 1, 1)
    ones = TTEngine.ones(
        [1], [c.get_numel(0) * (c.degrees[0] + 1)], torch.device(device), dtype
    )
    length = mm(ones, actual).to_dense()[0]
    assert abs(length - 1.0) < tol

    # =================================================================
    # Test assemble_loss_operator() method

    H = tt_assembler.assemble_loss_operator()
    assert H.is_tt
    H = H.as_tt()
    assert len(H) == 3
    assert H[0].shape[:-1] == (1, 8, 8)
    assert H[1].shape[1:-1] == (
        c.get_numel(0) + c.degrees[0],
        c.get_numel(0) + c.degrees[0],
    )
    assert H[2].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)
    for i in range(len(H) - 1):
        assert H[i].shape[-1] == H[i + 1].shape[0]

    # =================================================================
    # Test assemble_scatter_operator() method

    S = tt_assembler.assemble_scatter_operator()
    assert S.is_tt
    S = S.as_tt()
    assert len(S) == 3
    assert S[0].shape[:-1] == (1, 8, 8)
    assert S[1].shape[1:-1] == (
        c.get_numel(0) + c.degrees[0],
        c.get_numel(0) + c.degrees[0],
    )
    assert S[2].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)
    for i in range(len(S) - 1):
        assert S[i].shape[-1] == S[i + 1].shape[0]

    # =================================================================
    # Test assemble_fission_operator() method

    F = tt_assembler.assemble_fission_operator()
    assert not F.defined()


@pytest.mark.parametrize("device, dtype", test_params)
def test_1d_ellipse(device, dtype):
    # Tolerances for this test
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    dtol = 1e-2 if dtype == torch.float32 else 1e-8

    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get test XSs and quadrature set
    mat_label, xs_server, qset = filler_xs_and_quadrature(device, dtype, 1)

    # Get the exact curve from igakit
    c = Patch.from_igakit(
        ellipse((0, 0, 0), torch.pi), device=device, dtype=dtype, fill=mat_label
    )

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6 if dtype == torch.float32 else 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else 0
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True
    dense_assembler = DenseDIGAFirstOrderTransportBackend1D(c, qset, xs_server, config)
    tt_assembler = TTDIGAFirstOrderTransportBackend1D(c, qset, xs_server, config)

    # =================================================================
    # Test assemble_basis() method
    expected = c.evaluate_all_basis(dense_assembler.quad_points, 0)

    actual = dense_assembler.assemble_basis()
    assert actual.shape == (
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
    )
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    actual = tt_assembler.assemble_basis()
    assert actual[0].shape == (
        1,
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
        1,
    )
    torch.testing.assert_close(actual.to_dense(), expected, atol=tol, rtol=tol)

    # =================================================================
    # Test assemble_basis_ders() method
    expected = c.evaluate_all_basis(dense_assembler.quad_points, 1, False)

    actual = dense_assembler.assemble_basis_ders()
    assert actual.shape[1] == 2
    assert actual[:, 0, :].shape == (
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
    )
    assert actual[:, 1, :].shape == (
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
    )
    torch.testing.assert_close(actual[:, 0, :], expected[:, 0, :], atol=tol, rtol=tol)
    torch.testing.assert_close(actual[:, 1, :], expected[:, 1, :], atol=tol, rtol=tol)

    actual = tt_assembler.assemble_basis_ders()
    assert actual[0][0].shape == (
        1,
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
        1,
    )
    assert actual[1][0].shape == (
        1,
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
        1,
    )
    torch.testing.assert_close(
        actual[0].to_dense(), expected[:, 0, :], atol=tol, rtol=tol
    )
    torch.testing.assert_close(
        actual[1].to_dense(), expected[:, 1, :], atol=dtol, rtol=dtol
    )

    # =================================================================
    # Test assemble_jacobian() and assemble_jacobian_inverse()

    jac = tt_assembler.assemble_jacobian()
    jac_inv = tt_assembler.assemble_jacobian_inverse()

    actual = jac[0][0] * jac_inv[0][0]
    for i in range(1, 2):
        actual += jac[i][0] * jac_inv[i][0]

    # Check we only get ones (identity)
    torch.testing.assert_close(
        actual[0], torch.ones_like(actual[0]), atol=tol, rtol=tol
    )

    # =================================================================
    # Test assemble_scattering_kernel() method

    actual = tt_assembler.assemble_scattering_kernel()
    assert len(actual) == 2
    assert actual[0].shape[:-1] == (1, 8, 8)
    assert actual[1].shape[1:] == (3, 3, 1)
    assert actual[0].shape[-1] == actual[1].shape[0]

    # Apply angular weights to make this operator an integral
    actual = tt_assembler.apply_angular_weights(actual, [0])

    # Test isotropic flux case
    phi_iso = TTEngine.ones(actual.n_modes, actual.device, actual.dtype)

    # Check result is the 0th order group-to-group scattering XS tensor
    torch.testing.assert_close(
        mm(actual, phi_iso).to_dense().squeeze(),
        xs_server.get_material(mat_label)
        .scatter_gtg[0, :, :]
        .sum(0)
        .reshape((1, -1))
        .expand((8, 3)),
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_2d_circle(device, dtype):
    # Tolerances for this test
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    dtol = 1e-2 if dtype == torch.float32 else 1e-8

    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get test XSs and quadrature set
    mat_label, xs_server, qset = filler_xs_and_quadrature(device, dtype, 2)

    # Get the exact circle from igakit
    radius = 1.0
    c = Patch.from_igakit(
        refine(circle(radius), 10, 5), device=device, dtype=dtype, fill=mat_label
    )

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6 if dtype == torch.float32 else 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else 0
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True
    dense_assembler = DenseDIGAFirstOrderTransportBackend2D(c, qset, xs_server, config)
    tt_assembler = TTDIGAFirstOrderTransportBackend2D(c, qset, xs_server, config)

    # =================================================================
    # Test assemble_basis() method

    expected = c.evaluate_all_basis(dense_assembler.quad_points, 0)

    actual = dense_assembler.assemble_basis()
    assert actual.shape == (
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_numel(1) * (c.degrees[1] + 1),
        c.get_ctrlpts_size(0),
        c.get_ctrlpts_size(1),
    )
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    actual = tt_assembler.assemble_basis()
    assert actual[0].shape[:-1] == (
        1,
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
    )
    assert actual[1].shape[1:] == (
        c.get_numel(1) * (c.degrees[1] + 1),
        c.get_ctrlpts_size(1),
        1,
    )
    assert actual[0].shape[-1] == actual[1].shape[0]
    torch.testing.assert_close(actual.to_dense(), expected, atol=tol, rtol=tol)

    # =================================================================
    # Test assemble_basis_ders() method
    expected = c.evaluate_all_basis(dense_assembler.quad_points, 1, False)

    actual = dense_assembler.assemble_basis_ders()
    assert actual.shape[2] == c.ndim + 1
    for i in range(c.ndim + 1):
        assert actual[:, :, i, :, :].shape == (
            c.get_numel(0) * (c.degrees[0] + 1),
            c.get_numel(1) * (c.degrees[1] + 1),
            c.get_ctrlpts_size(0),
            c.get_ctrlpts_size(1),
        )
        torch.testing.assert_close(
            actual[:, :, i, :, :], expected[:, :, i, :, :], atol=tol, rtol=tol
        )

    actual = tt_assembler.assemble_basis_ders()
    for i in range(c.ndim + 1):
        assert actual[i][0].shape[:-1] == (
            1,
            c.get_numel(0) * (c.degrees[0] + 1),
            c.get_ctrlpts_size(0),
        )
        assert actual[i][1].shape[1:] == (
            c.get_numel(1) * (c.degrees[1] + 1),
            c.get_ctrlpts_size(1),
            1,
        )
        assert actual[i][0].shape[-1] == actual[i][1].shape[0]
        torch.testing.assert_close(
            actual[i].to_dense(),
            expected[:, :, i, :, :],
            atol=tol if i == 0 else dtol,
            rtol=tol if i == 0 else dtol,
        )

    # =================================================================
    # Test assemble_scattering_kernel() method

    actual = tt_assembler.assemble_scattering_kernel()
    assert len(actual) == 3
    assert actual[0].shape[:-1] == (1, 4, 4)
    assert actual[1].shape[1:-1] == (16, 16)
    assert actual[2].shape[1:] == (3, 3, 1)
    assert actual[0].shape[-1] == actual[1].shape[0]
    assert actual[1].shape[-1] == actual[2].shape[0]

    # =================================================================
    # Test assemble_jacobian() and assemble_jacobian_inverse()

    jac = tt_assembler.assemble_jacobian()
    jac_inv = tt_assembler.assemble_jacobian_inverse()

    # Multiply the two together to get the identity
    for i in range(2):
        for j in range(2):
            actual = jac_inv[0][i] * jac[0][j]

            for k in range(1, 2):
                actual += jac_inv[k][i] * jac[k][j]

            actual = actual.to_dense()

            # Check the result is an identity
            torch.testing.assert_close(
                actual,
                torch.ones_like(actual) if i == j else torch.zeros_like(actual),
                rtol=6 * dtol,
                atol=6 * dtol,
            )

    # =================================================================
    # Test assemble_integral_mapping() method

    # Check the integral evaluates to the area of the circle
    actual = tt_assembler.assemble_integral_mapping()
    assert len(actual) == 2
    assert actual[0].shape[:-1] == (1, c.get_numel(0) * (c.degrees[0] + 1), 1)
    assert actual[1].shape[1:] == (c.get_numel(1) * (c.degrees[1] + 1), 1, 1)
    assert actual[0].shape[-1] == actual[1].shape[0]
    ones = TTEngine.ones(
        [1, 1],
        [c.get_numel(0) * (c.degrees[0] + 1), c.get_numel(1) * (c.degrees[1] + 1)],
        torch.device(device),
        dtype,
    )
    area = mm(ones, actual).to_dense()[0]
    assert abs(area - torch.pi * radius**2) < tol

    # =================================================================
    # Test assemble_loss_operator() method

    H = tt_assembler.assemble_loss_operator()
    assert H.is_tt
    H = H.as_tt()
    assert len(H) == 5
    assert H[0].shape[:-1] == (1, 4, 4)
    assert H[1].shape[1:-1] == (16, 16)
    assert H[2].shape[1:-1] == (
        c.get_numel(0) + c.degrees[0],
        c.get_numel(0) + c.degrees[0],
    )
    assert H[3].shape[1:-1] == (
        c.get_numel(1) + c.degrees[1],
        c.get_numel(1) + c.degrees[1],
    )
    assert H[4].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)
    for i in range(len(H) - 1):
        assert H[i].shape[-1] == H[i + 1].shape[0]

    # =================================================================
    # Test assemble_scatter_operator() method

    S = tt_assembler.assemble_scatter_operator()
    assert S.is_tt
    S = S.as_tt()
    assert len(S) == 5
    assert S[0].shape[:-1] == (1, 4, 4)
    assert S[1].shape[1:-1] == (16, 16)
    assert S[2].shape[1:-1] == (
        c.get_numel(0) + c.degrees[0],
        c.get_numel(0) + c.degrees[0],
    )
    assert S[3].shape[1:-1] == (
        c.get_numel(1) + c.degrees[1],
        c.get_numel(1) + c.degrees[1],
    )
    assert S[4].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)
    for i in range(len(H) - 1):
        assert S[i].shape[-1] == S[i + 1].shape[0]

    # =================================================================
    # Test assemble_fission_operator() method

    F = tt_assembler.assemble_fission_operator()
    assert not F.defined()


@pytest.mark.parametrize("device, dtype", test_params)
def test_3d_cube(device, dtype):
    # Tolerances for this test
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    dtol = 1e-2 if dtype == torch.float32 else 1e-8

    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get test XSs and quadrature set
    mat_label, xs_server, qset = filler_xs_and_quadrature(device, dtype, 3)

    # Get the exact cube from igakit
    c = Patch.from_igakit(
        grid((3, 4, 5), 2), device=device, dtype=dtype, fill=mat_label
    )
    assert not c.is_rational()

    # Create NURBS version
    shape = list(c.ctrlptsw.shape)
    shape[-1] += 1
    ctrlptsw = torch.ones(shape, device=device, dtype=dtype)
    ctrlptsw[..., :-1] = c.ctrlptsw
    c_test = Patch(
        ctrlptsw,
        [BSplineBasis(b.knotvector, b.degree) for b in c.basis],
        is_rational=True,
    )
    assert c_test.is_rational()
    c_test.fill = mat_label
    c_test.finalize()

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-7 if dtype == torch.float32 else 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else 0
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True
    dense_assembler = DenseDIGAFirstOrderTransportBackend3D(
        c_test, qset, xs_server, config
    )
    tt_assembler = TTDIGAFirstOrderTransportBackend3D(c_test, qset, xs_server, config)

    # =================================================================
    # Test assemble_basis() method
    expected = c.evaluate_all_basis(dense_assembler.quad_points, 0)

    actual = dense_assembler.assemble_basis()
    assert actual.shape == (
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_numel(1) * (c.degrees[1] + 1),
        c.get_numel(2) * (c.degrees[2] + 1),
        c.get_ctrlpts_size(0),
        c.get_ctrlpts_size(1),
        c.get_ctrlpts_size(2),
    )
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    actual = tt_assembler.assemble_basis()
    assert actual[0].shape[:-1] == (
        1,
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
    )
    assert actual[1].shape[1:-1] == (
        c.get_numel(1) * (c.degrees[1] + 1),
        c.get_ctrlpts_size(1),
    )
    assert actual[2].shape[1:] == (
        c.get_numel(2) * (c.degrees[2] + 1),
        c.get_ctrlpts_size(2),
        1,
    )
    assert actual[0].shape[-1] == actual[1].shape[0]
    assert actual[1].shape[-1] == actual[2].shape[0]
    torch.testing.assert_close(actual.to_dense(), expected, atol=tol, rtol=tol)

    # =================================================================
    # Test assemble_basis_ders() method
    expected = c.evaluate_all_basis(dense_assembler.quad_points, 1, False)

    actual = dense_assembler.assemble_basis_ders()
    assert actual.shape[3] == c.ndim + 1
    for i in range(c.ndim + 1):
        assert actual[:, :, :, i, :, :, :].shape == (
            c.get_numel(0) * (c.degrees[0] + 1),
            c.get_numel(1) * (c.degrees[1] + 1),
            c.get_numel(2) * (c.degrees[2] + 1),
            c.get_ctrlpts_size(0),
            c.get_ctrlpts_size(1),
            c.get_ctrlpts_size(2),
        )
        torch.testing.assert_close(
            actual[:, :, :, i, :, :, :],
            expected[:, :, :, i, :, :, :],
            atol=tol,
            rtol=tol,
        )

    actual = tt_assembler.assemble_basis_ders()
    for i in range(c.ndim + 1):
        assert actual[i][0].shape[:-1] == (
            1,
            c.get_numel(0) * (c.degrees[0] + 1),
            c.get_ctrlpts_size(0),
        )
        assert actual[i][1].shape[1:-1] == (
            c.get_numel(1) * (c.degrees[1] + 1),
            c.get_ctrlpts_size(1),
        )
        assert actual[i][2].shape[1:] == (
            c.get_numel(2) * (c.degrees[2] + 1),
            c.get_ctrlpts_size(2),
            1,
        )
        assert actual[i][0].shape[-1] == actual[i][1].shape[0]
        assert actual[i][1].shape[-1] == actual[i][2].shape[0]
        torch.testing.assert_close(
            actual[i].to_dense(),
            expected[:, :, :, i, :, :, :],
            atol=tol if i == 0 else dtol,
            rtol=tol if i == 0 else dtol,
        )

    # =================================================================
    # Test assemble_scattering_kernel() method

    actual = tt_assembler.assemble_scattering_kernel()
    assert len(actual) == 3
    assert actual[0].shape[:-1] == (1, 8, 8)
    assert actual[1].shape[1:-1] == (16, 16)
    assert actual[2].shape[1:] == (3, 3, 1)
    assert actual[0].shape[-1] == actual[1].shape[0]
    assert actual[1].shape[-1] == actual[2].shape[0]

    # Apply angular weights to make this operator an integral
    actual = tt_assembler.apply_angular_weights(actual, [0, 1])

    # Test isotropic flux case
    phi_iso = TTEngine.ones(actual.n_modes, actual.device, actual.dtype)

    # Check result is the 0th order group-to-group scattering XS tensor
    torch.testing.assert_close(
        mm(actual, phi_iso).to_dense().squeeze(),
        xs_server.get_material(mat_label)
        .scatter_gtg[0, :, :]
        .sum(0)
        .reshape((1, 1, -1))
        .expand((8, 16, 3)),
        atol=tol,
        rtol=tol,
    )

    # =================================================================
    # Test assemble_jacobian() and assemble_jacobian_inverse()

    jac = tt_assembler.assemble_jacobian()
    jac_inv = tt_assembler.assemble_jacobian_inverse()

    # Multiply the two together to get the identity
    for i in range(3):
        for j in range(3):
            actual = jac_inv[0][i] * jac[0][j]

            for k in range(1, 3):
                actual += jac_inv[k][i] * jac[k][j]

            actual = actual.to_dense()

            # Check the result is an identity
            torch.testing.assert_close(
                actual,
                torch.ones_like(actual) if i == j else torch.zeros_like(actual),
                rtol=6 * dtol,
                atol=6 * dtol,
            )

    # =================================================================
    # Test assemble_integral_mapping() method

    # Check the integral evaluates to the volume of the cube
    actual = tt_assembler.assemble_integral_mapping()
    assert len(actual) == 3
    assert actual[0].shape[:-1] == (1, c.get_numel(0) * (c.degrees[0] + 1), 1)
    assert actual[1].shape[1:-1] == (c.get_numel(1) * (c.degrees[1] + 1), 1)
    assert actual[2].shape[1:] == (c.get_numel(2) * (c.degrees[2] + 1), 1, 1)
    assert actual[0].shape[-1] == actual[1].shape[0]
    assert actual[1].shape[-1] == actual[2].shape[0]
    ones = TTEngine.ones(
        [1, 1, 1],
        [
            c.get_numel(0) * (c.degrees[0] + 1),
            c.get_numel(1) * (c.degrees[1] + 1),
            c.get_numel(2) * (c.degrees[2] + 1),
        ],
        torch.device(device),
        dtype,
    )
    volume = mm(ones, actual).to_dense()[0]
    assert abs(volume - 1.0) < tol

    # =================================================================
    # Test assemble_loss_operator() method

    H = tt_assembler.assemble_loss_operator()
    assert H.is_tt
    H = H.as_tt()
    assert len(H) == 6
    assert H[0].shape[:-1] == (1, 8, 8)
    assert H[1].shape[1:-1] == (16, 16)
    assert H[2].shape[1:-1] == (
        c.get_numel(0) + c.degrees[0],
        c.get_numel(0) + c.degrees[0],
    )
    assert H[3].shape[1:-1] == (
        c.get_numel(1) + c.degrees[1],
        c.get_numel(1) + c.degrees[1],
    )
    assert H[4].shape[1:-1] == (
        c.get_numel(2) + c.degrees[2],
        c.get_numel(2) + c.degrees[2],
    )
    assert H[5].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)
    for i in range(len(H) - 1):
        assert H[i].shape[-1] == H[i + 1].shape[0]

    # =================================================================
    # Test assemble_scatter_operator() method

    S = tt_assembler.assemble_scatter_operator()
    assert S.is_tt
    S = S.as_tt()
    assert len(S) == 6
    assert S[0].shape[:-1] == (1, 8, 8)
    assert S[1].shape[1:-1] == (16, 16)
    assert S[2].shape[1:-1] == (
        c.get_numel(0) + c.degrees[0],
        c.get_numel(0) + c.degrees[0],
    )
    assert S[3].shape[1:-1] == (
        c.get_numel(1) + c.degrees[1],
        c.get_numel(1) + c.degrees[1],
    )
    assert S[4].shape[1:-1] == (
        c.get_numel(2) + c.degrees[2],
        c.get_numel(2) + c.degrees[2],
    )
    assert S[5].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)
    for i in range(len(H) - 1):
        assert S[i].shape[-1] == S[i + 1].shape[0]

    # =================================================================
    # Test assemble_fission_operator() method

    F = tt_assembler.assemble_fission_operator()
    assert not F.defined()


@pytest.mark.parametrize("device, dtype", test_params)
def test_3d_cylinder(device, dtype):
    # Tolerances for this test
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    dtol = 1e-2 if dtype == torch.float32 else 1e-8

    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get test XSs and quadrature set
    mat_label, xs_server, qset = filler_xs_and_quadrature(device, dtype, 3)

    # Get the exact cube from igakit
    radius = 1.0
    length = 2.0
    c = Patch.from_igakit(
        refine(extrude(circle(radius), length, 2), [5, 4, 3], 3),
        device=device,
        dtype=dtype,
        fill=mat_label,
    )
    assert c.is_rational()

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6 if dtype == torch.float32 else 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10) if dtype == torch.float32 else int(1e10)
    config.cross_jacobian_inverse = False if dtype == torch.float32 else True
    dense_assembler = DenseDIGAFirstOrderTransportBackend3D(c, qset, xs_server, config)
    tt_assembler = TTDIGAFirstOrderTransportBackend3D(c, qset, xs_server, config)

    # =================================================================
    # Test assemble_basis() method
    expected = c.evaluate_all_basis(dense_assembler.quad_points, 0)

    actual = dense_assembler.assemble_basis()
    assert actual.shape == (
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_numel(1) * (c.degrees[1] + 1),
        c.get_numel(2) * (c.degrees[2] + 1),
        c.get_ctrlpts_size(0),
        c.get_ctrlpts_size(1),
        c.get_ctrlpts_size(2),
    )
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)

    actual = tt_assembler.assemble_basis()
    assert actual[0].shape[:-1] == (
        1,
        c.get_numel(0) * (c.degrees[0] + 1),
        c.get_ctrlpts_size(0),
    )
    assert actual[1].shape[1:-1] == (
        c.get_numel(1) * (c.degrees[1] + 1),
        c.get_ctrlpts_size(1),
    )
    assert actual[2].shape[1:] == (
        c.get_numel(2) * (c.degrees[2] + 1),
        c.get_ctrlpts_size(2),
        1,
    )
    assert actual[0].shape[-1] == actual[1].shape[0]
    assert actual[1].shape[-1] == actual[2].shape[0]
    torch.testing.assert_close(actual.to_dense(), expected, atol=tol, rtol=tol)

    # =================================================================
    # Test assemble_basis_ders() method
    expected = c.evaluate_all_basis(dense_assembler.quad_points, 1, False)

    actual = dense_assembler.assemble_basis_ders()
    assert actual.shape[3] == c.ndim + 1
    for i in range(c.ndim + 1):
        assert actual[:, :, :, i, :, :, :].shape == (
            c.get_numel(0) * (c.degrees[0] + 1),
            c.get_numel(1) * (c.degrees[1] + 1),
            c.get_numel(2) * (c.degrees[2] + 1),
            c.get_ctrlpts_size(0),
            c.get_ctrlpts_size(1),
            c.get_ctrlpts_size(2),
        )
        torch.testing.assert_close(
            actual[:, :, :, i, :, :, :],
            expected[:, :, :, i, :, :, :],
            atol=tol,
            rtol=tol,
        )

    actual = tt_assembler.assemble_basis_ders()
    for i in range(c.ndim + 1):
        assert actual[i][0].shape[:-1] == (
            1,
            c.get_numel(0) * (c.degrees[0] + 1),
            c.get_ctrlpts_size(0),
        )
        assert actual[i][1].shape[1:-1] == (
            c.get_numel(1) * (c.degrees[1] + 1),
            c.get_ctrlpts_size(1),
        )
        assert actual[i][2].shape[1:] == (
            c.get_numel(2) * (c.degrees[2] + 1),
            c.get_ctrlpts_size(2),
            1,
        )
        assert actual[i][0].shape[-1] == actual[i][1].shape[0]
        assert actual[i][1].shape[-1] == actual[i][2].shape[0]
        torch.testing.assert_close(
            actual[i].to_dense(),
            expected[:, :, :, i, :, :, :],
            atol=tol if i == 0 else dtol,
            rtol=tol if i == 0 else dtol,
        )

    # =================================================================
    # Test assemble_scattering_kernel() method

    actual = tt_assembler.assemble_scattering_kernel()
    assert len(actual) == 3
    assert actual[0].shape[:-1] == (1, 8, 8)
    assert actual[1].shape[1:-1] == (16, 16)
    assert actual[2].shape[1:] == (3, 3, 1)
    assert actual[0].shape[-1] == actual[1].shape[0]
    assert actual[1].shape[-1] == actual[2].shape[0]

    # Apply angular weights to make this operator an integral
    actual = tt_assembler.apply_angular_weights(actual, [0, 1])

    # Test isotropic flux case
    phi_iso = TTEngine.ones(actual.n_modes, actual.device, actual.dtype)

    # Check result is the 0th order group-to-group scattering XS tensor
    torch.testing.assert_close(
        mm(actual, phi_iso).to_dense().squeeze(),
        xs_server.get_material(mat_label)
        .scatter_gtg[0, :, :]
        .sum(0)
        .reshape((1, 1, -1))
        .expand((8, 16, 3)),
        atol=tol,
        rtol=tol,
    )

    # =================================================================
    # Test assemble_jacobian() and assemble_jacobian_inverse()

    jac = tt_assembler.assemble_jacobian()
    jac_inv = tt_assembler.assemble_jacobian_inverse()

    # Multiply the two together to get the identity
    for i in range(3):
        for j in range(3):
            actual = jac_inv[0][i] * jac[0][j]

            for k in range(1, 3):
                actual += jac_inv[k][i] * jac[k][j]

            actual = actual.to_dense()

            # Check the result is an identity
            torch.testing.assert_close(
                actual,
                torch.ones_like(actual) if i == j else torch.zeros_like(actual),
                rtol=6 * dtol,
                atol=6 * dtol,
            )

    # =================================================================
    # Test assemble_integral_mapping() method

    # Check the integral evaluates to the volume of the cylinder
    actual = tt_assembler.assemble_integral_mapping()
    assert len(actual) == 3
    assert actual[0].shape[:-1] == (1, c.get_numel(0) * (c.degrees[0] + 1), 1)
    assert actual[1].shape[1:-1] == (c.get_numel(1) * (c.degrees[1] + 1), 1)
    assert actual[2].shape[1:] == (c.get_numel(2) * (c.degrees[2] + 1), 1, 1)
    assert actual[0].shape[-1] == actual[1].shape[0]
    assert actual[1].shape[-1] == actual[2].shape[0]
    ones = TTEngine.ones(
        [1, 1, 1],
        [
            c.get_numel(0) * (c.degrees[0] + 1),
            c.get_numel(1) * (c.degrees[1] + 1),
            c.get_numel(2) * (c.degrees[2] + 1),
        ],
        torch.device(device),
        dtype,
    )
    volume = mm(ones, actual).to_dense()[0]
    assert abs(volume - torch.pi * radius**2 * length) < tol * 5

    # =================================================================
    # Test assemble_loss_operator() method

    H = tt_assembler.assemble_loss_operator()
    assert H.is_tt
    H = H.as_tt()
    assert len(H) == 6
    assert H[0].shape[:-1] == (1, 8, 8)
    assert H[1].shape[1:-1] == (16, 16)
    assert H[2].shape[1:-1] == (
        c.get_numel(0) + c.degrees[0],
        c.get_numel(0) + c.degrees[0],
    )
    assert H[3].shape[1:-1] == (
        c.get_numel(1) + c.degrees[1],
        c.get_numel(1) + c.degrees[1],
    )
    assert H[4].shape[1:-1] == (
        c.get_numel(2) + c.degrees[2],
        c.get_numel(2) + c.degrees[2],
    )
    assert H[5].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)
    for i in range(len(H) - 1):
        assert H[i].shape[-1] == H[i + 1].shape[0]

    # =================================================================
    # Test assemble_scatter_operator() method

    S = tt_assembler.assemble_scatter_operator()
    assert S.is_tt
    S = S.as_tt()
    assert len(S) == 6
    assert S[0].shape[:-1] == (1, 8, 8)
    assert S[1].shape[1:-1] == (16, 16)
    assert S[2].shape[1:-1] == (
        c.get_numel(0) + c.degrees[0],
        c.get_numel(0) + c.degrees[0],
    )
    assert S[3].shape[1:-1] == (
        c.get_numel(1) + c.degrees[1],
        c.get_numel(1) + c.degrees[1],
    )
    assert S[4].shape[1:-1] == (
        c.get_numel(2) + c.degrees[2],
        c.get_numel(2) + c.degrees[2],
    )
    assert S[5].shape[1:] == (xs_server.num_groups, xs_server.num_groups, 1)
    for i in range(len(H) - 1):
        assert S[i].shape[-1] == S[i + 1].shape[0]

    # =================================================================
    # Test assemble_fission_operator() method

    F = tt_assembler.assemble_fission_operator()
    assert not F.defined()
