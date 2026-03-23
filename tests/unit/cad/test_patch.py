import pytest
import torch
import numpy as np
from igakit.cad import refine

from ttnte.cad.surfaces import circle
from ttnte.cad import Patch
from ttnte.physics import BoundaryType

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    # ("cuda", torch.float32),
    # ("cuda", torch.float64),
]


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
