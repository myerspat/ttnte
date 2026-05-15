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
    torch.testing.assert_close(c.ctrlpts, ctrlpts)
    torch.testing.assert_close(c.weights, weights)
    torch.testing.assert_close(
        c.ctrlptsw, torch.tensor(c_exact.control, device=device, dtype=dtype)
    )

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
        torch.tensor(c_exact(u, v).reshape((-1, 3)), device=device, dtype=dtype),
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
        torch.tensor(c_exact(u, v), device=device, dtype=dtype),
    )


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
    ctrlpts = torch.zeros((3, 2, 2), device=device, dtype=dtype)
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
