import pytest
import torch

from ttnte.linalg import TTState

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]


@pytest.mark.parametrize("device, dtype", test_params)
def test_initialize(device, dtype):
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cores = [
        torch.ones(i * 10, device=device, dtype=dtype).reshape((1, -1, 1))
        for i in range(1, 4)
    ]
    name = "tt operator"

    tt = TTState(cores, name)

    assert tt.label.to_string() == "tt operator"
    assert (
        tt.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert tt.dtype == dtype
    assert tt.numel == 60

    for i in range(1, 4):
        torch.testing.assert_close(
            tt.cores[i - 1],
            torch.ones(i * 10, device=device, dtype=dtype).reshape((1, -1, 1, 1)),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_methods(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cores = [
        torch.ones(i * 10, device="cpu", dtype=torch.float64).reshape((1, -1, 1))
        for i in range(1, 4)
    ]

    tt0 = TTState(cores)

    # Send to GPU as dtype
    tt1 = tt0.to(torch.device("cuda", 0), dtype)

    # Check the getters
    assert tt0.device == torch.device("cpu")
    assert tt0.dtype == torch.float64
    assert tt1.device == torch.device("cuda", 0)
    assert tt1.dtype == dtype

    # Check individual tensors
    for tt0_core, tt1_core in zip(tt0.cores, tt1.cores):
        assert tt0_core.device == torch.device("cpu")
        assert tt0_core.dtype == torch.float64
        assert tt1_core.device == torch.device("cuda", 0)
        assert tt1_core.dtype == dtype

    # Test in-place version
    tt0.to_(torch.device("cuda", 0), dtype)

    # Check the getters
    assert tt0.device == torch.device("cuda", 0)
    assert tt0.dtype == dtype

    # Check individual tensors
    for tt0_core in tt0.cores:
        assert tt0_core.device == torch.device("cuda", 0)
        assert tt0_core.dtype == dtype
