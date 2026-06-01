from itertools import product

import pytest
import torch
import numpy as np
from igakit.cad import refine

from ttnte.cad.surfaces import circle
from ttnte.cad import Patch, BSplineBasis
from ttnte.physics import BoundaryType

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    # ("cuda", torch.float32),
    # ("cuda", torch.float64),
]


def _expected_local_basis(patch, coords):
    basis_values = []
    spans = []

    for axis, values in enumerate(coords):
        basis = patch.basis[axis]
        axis_spans = basis.find_spans(values)
        spans.append(axis_spans)
        basis_values.append(basis.evaluate(values, axis_spans))

    result = torch.einsum("ip,jq->ijpq", basis_values[0], basis_values[1])

    if patch.is_rational():
        degree_u = patch.basis[0].degree
        degree_v = patch.basis[1].degree
        weights = patch.weights
        local_weights = torch.stack(
            [
                torch.stack(
                    [
                        weights[
                            spans[0][i] - degree_u : spans[0][i] + 1,
                            spans[1][j] - degree_v : spans[1][j] + 1,
                        ]
                        for j in range(coords[1].numel())
                    ],
                    dim=0,
                )
                for i in range(coords[0].numel())
            ],
            dim=0,
        )
        result = result * local_weights
        result = result / result.sum(dim=(-2, -1), keepdim=True)

    return result


@pytest.mark.parametrize("device, dtype", test_params)
def test_patch(device, dtype):
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create test circle
    c_exact = refine(circle(2), factor=[3, 1], degree=[2, 3])

    # Control points and weights
    ctrlpts = torch.tensor(
        c_exact.control[..., :-1] / c_exact.control[..., [-1]],
        device=device,
        dtype=dtype,
    )
    weights = torch.tensor(c_exact.control[..., -1], device=device, dtype=dtype)
    device = ctrlpts.device
    dtype = ctrlpts.dtype

    # Create NURBS
    c = Patch.from_igakit(c_exact, device=device, dtype=dtype)

    # Checks
    assert c.is_finalized() == True
    assert c.is_rational() == True
    assert c.device == device
    assert c.dtype == dtype
    torch.testing.assert_close(c.ctrlpts, ctrlpts[..., :-1])
    torch.testing.assert_close(c.weights, weights)

    for i, b in enumerate(c.basis):
        assert b.degree == c_exact.degree[i]
        torch.testing.assert_close(
            b.knotvector, torch.tensor(c_exact.knots[i], device=device, dtype=dtype)
        )

    for i, dim, is_upper in zip([0, 1, 2, 3], [0, 0, 1, 1], [False, True, False, True]):
        binfo = c.get_boundary_info(dim, is_upper)
        assert binfo.fid == i
        assert binfo.type == BoundaryType.UNKNOWN
        assert binfo.connections == []

    # Test evaluate
    u = np.linspace(0, 1, 5)
    v = np.linspace(0, 1, 5)
    local_coords = np.stack(
        [x.flatten() for x in np.meshgrid(u, v, indexing="ij")], axis=-1
    )

    torch.testing.assert_close(
        # Actual
        c.evaluate(
            torch.tensor(local_coords, device=device, dtype=dtype),
        ),
        # Expected
        torch.tensor(
            c_exact(u, v)[..., :-1].reshape((-1, 2)), device=device, dtype=dtype
        ),
    )
    torch.testing.assert_close(
        # Actual
        c.evaluate(
            [
                torch.tensor(u, device=device, dtype=dtype),
                torch.tensor(v, device=device, dtype=dtype),
            ],
        ),
        # Expected
        torch.tensor(c_exact(u, v)[..., :-1], device=device, dtype=dtype),
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_evaluate_all_basis(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    atol = 1e-3 if dtype == torch.float32 else 5e-12
    rtol = 1e-3 if dtype == torch.float32 else 5e-12

    # ======================================================
    # Test the partition of unity
    c_exact = refine(circle(2), factor=[3, 1], degree=[2, 3])
    c_exact = c_exact.insert(0, c_exact.knots[0][4], 1)

    # Create NURBS
    c = Patch.from_igakit(c_exact, device=device, dtype=dtype)

    u = torch.linspace(0, 1, 4, device=device, dtype=dtype)
    v = torch.linspace(0, 1, 3, device=device, dtype=dtype)

    actual = c.evaluate_all_basis([u, v], 3, True)
    assert actual.shape == (4, 3, 3, 4, 6, 4)

    actual = torch.sum(actual, (-2, -1))

    for i, j in product(range(3), range(4)):
        actual_ij = actual[..., i, j]

        if i == 0 and j == 0:
            torch.testing.assert_close(
                actual_ij, torch.ones_like(actual_ij), atol=atol, rtol=rtol
            )
        else:
            torch.testing.assert_close(
                actual_ij, torch.zeros_like(actual_ij), atol=atol, rtol=rtol
            )

    # Case when we don't want cross derivatives
    actual = c.evaluate_all_basis([u, v], 3, False)

    # K_flat = 1 (zeroth) + min(3, degree_u) + min(3, degree_v)
    # Degree of u is 2 -> min(3, 2) = 2 pure u derivatives
    # Degree of v is 3 -> min(3, 3) = 3 pure v derivatives
    # K_flat = 1 + 2 + 3 = 6
    assert actual.shape == (4, 3, 6, 6, 4)

    actual = torch.sum(actual, (-2, -1))

    for k in range(6):
        actual_k = actual[..., k]

        if k == 0:  # Zeroth derivative should sum to 1
            torch.testing.assert_close(
                actual_k, torch.ones_like(actual_k), atol=atol, rtol=rtol
            )
        else:  # All pure higher-order derivatives should sum to 0
            torch.testing.assert_close(
                actual_k, torch.zeros_like(actual_k), atol=atol, rtol=rtol
            )

    # ======================================================
    # Test analytic B-spline example

    # Create test B-Splines
    basis = [
        BSplineBasis(torch.tensor([0, 0, 1, 1], device=device, dtype=dtype), 1),
        BSplineBasis(torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=dtype), 2),
    ]
    ctrlpts = torch.cat(
        [
            ctrl.unsqueeze(-1)
            for ctrl in torch.meshgrid(
                torch.linspace(0, 1, 2), torch.linspace(0, 1, 3), indexing="ij"
            )
        ],
        -1,
    ).to(device, dtype)
    c = Patch(ctrlpts, basis, False)
    c.finalize()

    # Test points
    u = torch.linspace(0, 1, 4, device=device, dtype=dtype)
    v = torch.linspace(0, 1, 5, device=device, dtype=dtype)

    # Evaluate up to the 1st derivative (which gives us d^2/dudv)
    # Expected shape: (u_pts=4, v_pts=5, u_deriv=2, v_deriv=2, u_basis=2, v_basis=3)
    actual = c.evaluate_all_basis([u, v], 1, True)

    # U-direction (Degree 1)
    N_u = torch.stack([1.0 - u, u], dim=-1)  # Shape: (4, 2)
    dN_u = torch.stack([-torch.ones_like(u), torch.ones_like(u)], dim=-1)

    # V-direction (Degree 2)
    M_v = torch.stack(
        [(1.0 - v) ** 2, 2.0 * v * (1.0 - v), v**2], dim=-1
    )  # Shape: (5, 3)
    dM_v = torch.stack([2.0 * v - 2.0, 2.0 - 4.0 * v, 2.0 * v], dim=-1)

    # Updated to layout: (u_pts, v_pts, u_deriv, v_deriv, u_basis, v_basis)
    expected = torch.zeros((4, 5, 2, 2, 2, 3), device=device, dtype=dtype)

    # Reshape for broadcasting (this part stays exactly the same,
    # it creates the (4, 5, 2, 3) block we will inject into the slices)
    # U components: (4, 1, 2, 1) -> unsqueeze v_pts and v_basis dims
    # V components: (1, 5, 1, 3) -> unsqueeze u_pts and u_basis dims
    N_u_b = N_u.view(4, 1, 2, 1)
    dN_u_b = dN_u.view(4, 1, 2, 1)

    M_v_b = M_v.view(1, 5, 1, 3)
    dM_v_b = dM_v.view(1, 5, 1, 3)

    # Now assign to the proper derivative dimensions (middle dims 2 and 3)
    # Base values (0th deriv u, 0th deriv v)
    expected[:, :, 0, 0, :, :] = N_u_b * M_v_b

    # d/du (1st deriv u, 0th deriv v)
    expected[:, :, 1, 0, :, :] = dN_u_b * M_v_b

    # d/dv (0th deriv u, 1st deriv v)
    expected[:, :, 0, 1, :, :] = N_u_b * dM_v_b

    # d^2 / dudv (1st deriv u, 1st deriv v) -- THE CROSS DERIVATIVE!
    expected[:, :, 1, 1, :, :] = dN_u_b * dM_v_b

    # Tight tolerance because this is a non-rational pure polynomial
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    # Case when we want no cross derivatives
    # Evaluate up to the 1st derivative
    actual = c.evaluate_all_basis([u, v], 1, False)

    # Orders are min(1, 1) = 1 for u, and min(1, 2) = 1 for v.
    # K_flat = 1 (base) + 1 (u) + 1 (v) = 3
    # Expected shape: (u_pts=4, v_pts=5, K_flat=3, u_basis=2, v_basis=3)
    assert actual.shape == (4, 5, 3, 2, 3)

    # U-direction (Degree 1)
    N_u = torch.stack([1.0 - u, u], dim=-1)  # Shape: (4, 2)
    dN_u = torch.stack([-torch.ones_like(u), torch.ones_like(u)], dim=-1)

    # V-direction (Degree 2)
    M_v = torch.stack(
        [(1.0 - v) ** 2, 2.0 * v * (1.0 - v), v**2], dim=-1
    )  # Shape: (5, 3)
    dM_v = torch.stack([2.0 * v - 2.0, 2.0 - 4.0 * v, 2.0 * v], dim=-1)

    expected = torch.zeros((4, 5, 3, 2, 3), device=device, dtype=dtype)

    # Reshape for broadcasting
    N_u_b = N_u.view(4, 1, 2, 1)
    dN_u_b = dN_u.view(4, 1, 2, 1)

    M_v_b = M_v.view(1, 5, 1, 3)
    dM_v_b = dM_v.view(1, 5, 1, 3)

    # Now assign to the proper K_flat indices (dimension 2)

    # Index 0: Base values (0th deriv u, 0th deriv v)
    expected[:, :, 0, :, :] = N_u_b * M_v_b

    # Index 1: d/du (1st deriv u, 0th deriv v)
    expected[:, :, 1, :, :] = dN_u_b * M_v_b

    # Index 2: d/dv (0th deriv u, 1st deriv v)
    expected[:, :, 2, :, :] = N_u_b * dM_v_b

    # Tight tolerance because this is a non-rational pure polynomial
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_evaluate_basis(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    c_exact = refine(circle(2), factor=[3, 1], degree=[2, 3])
    c_exact = c_exact.insert(0, c_exact.knots[0][4], 1)

    # Create NURBS
    c = Patch.from_igakit(c_exact, device=device, dtype=dtype)

    u = torch.linspace(0, 1, 4, device=device, dtype=dtype)
    v = torch.linspace(0, 1, 3, device=device, dtype=dtype)

    actual = c.evaluate_basis([u, v])
    expected = _expected_local_basis(c, [u, v])

    assert actual.shape == (
        u.numel(),
        v.numel(),
        c.basis[0].degree + 1,
        c.basis[1].degree + 1,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual.sum(dim=(-2, -1)), torch.ones_like(actual[..., 0, 0])
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_evaluate_basis_checks(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    basis = [
        BSplineBasis(torch.tensor([0, 0, 0, 1, 1, 1], device=device, dtype=dtype), 2),
        BSplineBasis(torch.tensor([0, 0, 1, 1], device=device, dtype=dtype), 1),
    ]
    ctrlpts = torch.cat(
        [
            ctrl.unsqueeze(-1)
            for ctrl in torch.meshgrid(
                torch.linspace(0, 1, 3), torch.linspace(0, 1, 2), indexing="ij"
            )
        ],
        -1,
    ).to(device, dtype)
    patch = Patch(ctrlpts, basis)
    patch.finalize()

    with pytest.raises(RuntimeError, match="1D"):
        patch.evaluate_basis(
            [
                torch.zeros((2, 2), device=device, dtype=dtype),
                torch.zeros(2, device=device, dtype=dtype),
            ]
        )

    mismatch_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    with pytest.raises(RuntimeError, match="same dtype and device"):
        patch.evaluate_basis(
            [
                torch.zeros(2, device=device, dtype=dtype),
                torch.zeros(2, device=device, dtype=mismatch_dtype),
            ]
        )


# =======================================
# Test Knot Refinement Methods
# =======================================

# =======================================
# Test Knot Insertion Methods


# Test Knot Insertion in First Parametric Dimension
@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_knot_insert_dir0(device, dtype):
    """Non-in-place knot insertion along direction 0 preserves geometry."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create circle to test insertion methods on
    c_exact = refine(circle(2), factor=[3, 1], degree=[2, 3])
    c = Patch.from_igakit(c_exact, device=device, dtype=dtype)

    new_knot = 0.25
    original_kv0 = torch.tensor(c_exact.knots[0], device=device, dtype=dtype)
    original_kv1 = torch.tensor(c_exact.knots[1], device=device, dtype=dtype)
    c_ref = c_exact.insert(0, new_knot, 1)

    new_knots = [
        torch.tensor([new_knot], device=device, dtype=dtype),
        torch.tensor([], device=device, dtype=dtype),
    ]
    c_inserted = c.knot_insert(new_knots)

    # Original patch is unchanged
    torch.testing.assert_close(c.basis[0].knotvector, original_kv0)

    # Knot vector updated correctly
    torch.testing.assert_close(
        c_inserted.basis[0].knotvector,
        torch.tensor(c_ref.knots[0], device=device, dtype=dtype),
    )
    # Direction 1 is unchanged
    torch.testing.assert_close(c_inserted.basis[1].knotvector, original_kv1)

    # Control points match igakit reference
    torch.testing.assert_close(
        c_inserted.ctrlptsw,
        torch.tensor(c_ref.control, device=device, dtype=dtype),
    )

    # Check geometry preservation
    u = torch.linspace(0, 1, 7, device=device, dtype=dtype)
    v = torch.linspace(0, 1, 7, device=device, dtype=dtype)
    torch.testing.assert_close(c.evaluate([u, v]), c_inserted.evaluate([u, v]))


# Test Knot Insertion in Second Parametric Dimension
@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_knot_insert_dir1(device, dtype):
    """Non-in-place knot insertion along direction 1 preserves geometry."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create circle to test insertion methods on
    c_exact = refine(circle(2), factor=[3, 1], degree=[2, 3])
    c = Patch.from_igakit(c_exact, device=device, dtype=dtype)

    new_knot = 0.6
    c_ref = c_exact.insert(1, new_knot, 1)

    new_knots = [
        torch.tensor([], device=device, dtype=dtype),
        torch.tensor([new_knot], device=device, dtype=dtype),
    ]
    c_inserted = c.knot_insert(new_knots)

    # Direction 1 is unchanged
    torch.testing.assert_close(
        c_inserted.basis[1].knotvector,
        torch.tensor(c_ref.knots[1], device=device, dtype=dtype),
    )

    # Control points match igakit reference
    torch.testing.assert_close(
        c_inserted.ctrlptsw,
        torch.tensor(c_ref.control, device=device, dtype=dtype),
    )

    # Check geometry preservation
    u = torch.linspace(0, 1, 7, device=device, dtype=dtype)
    v = torch.linspace(0, 1, 7, device=device, dtype=dtype)
    torch.testing.assert_close(c.evaluate([u, v]), c_inserted.evaluate([u, v]))


# Test Multiple Insertion of The Same Knots
@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_knot_insert_reps(device, dtype):
    """Reps > 1 inserts the same knots multiple times."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create circle to test insertion methods on
    c_exact = refine(circle(2), factor=[3, 1], degree=[2, 3])
    c = Patch.from_igakit(c_exact, device=device, dtype=dtype)

    new_knot = 0.25
    reps = 2
    # igakit reference: insert the same knot `reps` times
    c_ref = c_exact
    for _ in range(reps):
        c_ref = c_ref.insert(0, new_knot, 1)

    new_knots = [
        torch.tensor([new_knot], device=device, dtype=dtype),
        torch.tensor([], device=device, dtype=dtype),
    ]
    c_inserted = c.knot_insert(new_knots, reps=reps)

    # Knots in first dimension and control points match igakit reference
    torch.testing.assert_close(
        c_inserted.basis[0].knotvector,
        torch.tensor(c_ref.knots[0], device=device, dtype=dtype),
    )
    torch.testing.assert_close(
        c_inserted.ctrlptsw,
        torch.tensor(c_ref.control, device=device, dtype=dtype),
    )

    # Check geometry preservation
    u = torch.linspace(0, 1, 7, device=device, dtype=dtype)
    v = torch.linspace(0, 1, 7, device=device, dtype=dtype)
    torch.testing.assert_close(c.evaluate([u, v]), c_inserted.evaluate([u, v]))


# Test Knot Insertion In-Place Method
@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_knot_insert_inplace(device, dtype):
    """In-place knot insertion matches the non-in-place result."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create circle to test insertion methods on
    c_exact = refine(circle(2), factor=[3, 1], degree=[2, 3])

    new_knots = [
        torch.tensor([0.25], device=device, dtype=dtype),
        torch.tensor([], device=device, dtype=dtype),
    ]

    c_out_of_place = Patch.from_igakit(c_exact, device=device, dtype=dtype)
    c_inplace = Patch.from_igakit(c_exact, device=device, dtype=dtype)

    result = c_out_of_place.knot_insert(new_knots)
    c_inplace.knot_insert_(new_knots)

    # First Dimension knot vectors match igakit reference
    torch.testing.assert_close(
        c_inplace.basis[0].knotvector,
        result.basis[0].knotvector,
    )
    torch.testing.assert_close(c_inplace.ctrlptsw, result.ctrlptsw)

    # Check geometry preservation
    u = torch.linspace(0, 1, 7, device=device, dtype=dtype)
    v = torch.linspace(0, 1, 7, device=device, dtype=dtype)
    torch.testing.assert_close(
        c_inplace.evaluate([u, v]),
        result.evaluate([u, v]),
    )
