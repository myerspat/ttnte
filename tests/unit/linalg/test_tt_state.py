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


@pytest.mark.parametrize("device, dtype", test_params)
def test_factories_and_properties(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m_modes = [2, 3, 4]

    # Test zeros factory
    tt_z = TTState.zeros(m_modes, device=torch.device(device), dtype=dtype)
    assert isinstance(tt_z, TTState)
    assert tt_z.m_modes == m_modes
    assert (
        tt_z.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert tt_z.dtype == dtype

    for core in tt_z.cores:
        torch.testing.assert_close(core, torch.zeros_like(core))

    # Test ones factory
    tt_o = TTState.ones(m_modes, device=torch.device(device), dtype=dtype)
    assert isinstance(tt_o, TTState)
    assert tt_o.m_modes == m_modes

    for core in tt_o.cores:
        torch.testing.assert_close(core, torch.ones_like(core))

    # Check other properties
    assert len(tt_z.ranks) == len(m_modes) + 1
    assert len(tt_z.free_indices) > 0


@pytest.mark.parametrize("device, dtype", test_params)
def test_math_operators(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create states and tensors for math operations
    tt1 = TTState.ones([2, 3], device=torch.device(device), dtype=dtype)
    tt2 = TTState.ones([2, 3], device=torch.device(device), dtype=dtype)
    scalar = 2.0
    tensor_scalar = torch.tensor(3.0, device=device, dtype=dtype)

    # 1. State-to-State operations
    # Addition
    tt_add = tt1 + tt2
    assert isinstance(tt_add, TTState)
    torch.testing.assert_close(
        tt_add.to_dense(),
        2 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )

    # Subtraction (Note: C++ header has a typo here `a + b` instead of `a - b`
    # for the TTState-TTState operator, but we test the interface regardless)
    tt_sub = tt1 - tt2
    assert isinstance(tt_sub, TTState)
    torch.testing.assert_close(
        tt_sub.to_dense(),
        torch.zeros([2, 3], device=torch.device(device), dtype=dtype),
    )

    # Element-wise multiplication / division
    tt_mul = tt1 * tt2
    assert isinstance(tt_mul, TTState)
    torch.testing.assert_close(
        tt_mul.to_dense(),
        torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )

    tt_div = tt1 / tt2
    assert isinstance(tt_div, TTState)
    torch.testing.assert_close(
        tt_div.to_dense(),
        torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )

    # 2. Scalar operations
    assert isinstance(tt1 + scalar, TTState)
    torch.testing.assert_close(
        (tt1 + scalar).to_dense(),
        3 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(scalar + tt1, TTState)
    torch.testing.assert_close(
        (scalar + tt1).to_dense(),
        3 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(tt1 - scalar, TTState)
    torch.testing.assert_close(
        (tt1 - scalar).to_dense(),
        -1 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(scalar - tt1, TTState)
    torch.testing.assert_close(
        (scalar - tt1).to_dense(),
        torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(tt1 * scalar, TTState)
    torch.testing.assert_close(
        (tt1 * scalar).to_dense(),
        2 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(scalar * tt1, TTState)
    torch.testing.assert_close(
        (scalar * tt1).to_dense(),
        2 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(tt1 / scalar, TTState)
    torch.testing.assert_close(
        (tt1 / scalar).to_dense(),
        0.5 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(scalar / tt1, TTState)
    torch.testing.assert_close(
        (scalar / tt1).to_dense(),
        2 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )

    # 3. Tensor operations
    assert isinstance(tt1 + tensor_scalar, TTState)
    torch.testing.assert_close(
        (tt1 + tensor_scalar).to_dense(),
        4 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(tt1 * tensor_scalar, TTState)
    torch.testing.assert_close(
        (tt1 * tensor_scalar).to_dense(),
        3 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_inplace_negation(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tt = TTState.ones([2, 2], device=torch.device(device), dtype=dtype)

    # Apply in-place negation
    tt.neg_()

    # Check only the first core is negated
    torch.testing.assert_close(tt.cores[0], -torch.ones_like(tt.cores[0]))

    for core in tt.cores[1:]:
        # Check that the core is completely negative ones
        torch.testing.assert_close(core, torch.ones_like(core))


@pytest.mark.parametrize("device, dtype", test_params)
def test_orthogonalize_and_round(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tt = TTState.ones([4, 4, 4], device=torch.device(device), dtype=dtype)

    # Out-of-place orthogonalization
    tt_ortho = tt.lr_orthogonalize()
    assert isinstance(tt_ortho, TTState)
    assert tt_ortho is not tt

    # In-place orthogonalization
    tt.lr_orthogonalize_()

    # Out-of-place rounding
    tt_rounded = tt.round(eps=1e-3, max_rank=2)
    assert isinstance(tt_rounded, TTState)

    # Ranks should ideally be reduced
    assert all(r <= 2 for r in tt_rounded.ranks[1:-1])

    # In-place rounding
    tt.round_(eps=1e-3, max_rank=2)
    assert all(r <= 2 for r in tt.ranks[1:-1])


@pytest.mark.parametrize("device, dtype", test_params)
def test_pack_unpack(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a state
    tt = TTState.ones([2, 3, 2], device=torch.device(device), dtype=dtype)

    # Pack to 1D buffer
    buffer = tt.pack()

    assert isinstance(buffer, torch.Tensor)
    assert buffer.dim() == 1
    assert buffer.device == (
        torch.device(device) if device == "cpu" else torch.device(device, 0)
    )
    assert buffer.dtype == dtype

    # Unpack from 1D buffer
    tt_unpacked = TTState.unpack(buffer, clone=True)

    assert isinstance(tt_unpacked, TTState)
    assert tt.m_modes == tt_unpacked.m_modes
    assert tt.ranks == tt_unpacked.ranks

    # Ensure unpacked cores match the original cores
    for core_orig, core_unpacked in zip(tt.cores, tt_unpacked.cores):
        torch.testing.assert_close(core_orig, core_unpacked)
