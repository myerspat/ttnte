import math
import pytest
import torch

from ttnte.math import ProductQuadrature, sph_harm, HarmonicComponent

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    # ("cuda", torch.float32),
    # ("cuda", torch.float64),
]


@pytest.mark.parametrize("device, dtype", test_params)
def test_sph_harm(device, dtype):
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get a quadrature set
    qset = ProductQuadrature.chebyshev_legendre(4)
    qset.to_(torch.device(device), dtype)
    mu, gamma = [qset.get_points()[:, i] for i in range(2)]

    # Test l = 0 and m = 0
    y = sph_harm(0, 0, mu, gamma)
    assert (y[:, 0] == 1).all()
    assert (y[:, 1] == 0).all()

    # Test l = 2 and m = 0
    y = sph_harm(
        2 * torch.ones_like(mu, device=device, dtype=torch.int32),
        torch.zeros_like(mu, device=device, dtype=torch.int32),
        mu,
        gamma,
    )
    torch.testing.assert_close(y[:, 0], math.sqrt(5 / 4) * (3 * mu**2 - 1))
    assert (y[:, 1] == 0).all()

    # Test l = 2 and m = 1
    y = sph_harm(
        2 * torch.ones_like(mu, device=device, dtype=torch.int32),
        torch.ones_like(mu, device=device, dtype=torch.int32),
        mu,
        gamma,
        HarmonicComponent.ODD,
    )
    torch.testing.assert_close(
        y[:, 0], math.sqrt(15 / 2) * mu * torch.sqrt(1 - mu**2) * torch.sin(gamma)
    )

    # Test l = 2 and m = 2
    y = sph_harm(
        2 * torch.ones_like(mu, device=device, dtype=torch.int32),
        2 * torch.ones_like(mu, device=device, dtype=torch.int32),
        mu,
        gamma,
        torch.zeros_like(mu, device=device, dtype=torch.bool),
    )
    torch.testing.assert_close(
        y, math.sqrt(15 / 8) * (1 - mu**2) * torch.cos(2 * gamma)
    )
    y = sph_harm(
        2 * torch.ones_like(mu, device=device, dtype=torch.int32),
        2 * torch.ones_like(mu, device=device, dtype=torch.int32),
        mu,
        gamma,
        torch.ones_like(mu, device=device, dtype=torch.bool),
    )
    torch.testing.assert_close(
        y, math.sqrt(15 / 8) * (1 - mu**2) * torch.sin(2 * gamma)
    )
