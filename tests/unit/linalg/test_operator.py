import pytest
import torch

from ttnte.linalg import Operator, TTEngine

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
        torch.ones((i * 10, i * 10), device=device, dtype=dtype).reshape(
            (1, i * 10, i * 10, 1)
        )
        for i in range(1, 4)
    ]
    name = "tt operator"

    tt = Operator(TTEngine(cores), name)

    assert tt.is_tt
    assert tt.label.to_string() == "tt operator"
    assert (
        tt.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert tt.dtype == dtype
    assert tt.numel == 1400

    for i in range(1, 4):
        torch.testing.assert_close(
            tt.as_tt().cores[i - 1],
            torch.ones((i * 10, i * 10), device=device, dtype=dtype).reshape(
                (1, i * 10, i * 10, 1)
            ),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_methods(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cores = [
        torch.ones((i * 10, i * 10), device="cpu", dtype=torch.float64).reshape(
            (1, i * 10, i * 10, 1)
        )
        for i in range(1, 4)
    ]

    tt0 = Operator(TTEngine(cores))
    assert tt0.is_tt

    # Send to GPU as dtype
    tt1 = tt0.to(torch.device("cuda", 0), dtype)
    assert tt1.is_tt

    # Check the getters
    assert tt0.device == torch.device("cpu")
    assert tt0.dtype == torch.float64
    assert tt1.device == torch.device("cuda", 0)
    assert tt1.dtype == dtype

    # Check individual tensors
    for tt0_core, tt1_core in zip(tt0.as_tt().cores, tt1.as_tt().cores):
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
    for tt0_core in tt0.as_tt().cores:
        assert tt0_core.device == torch.device("cuda", 0)
        assert tt0_core.dtype == dtype


@pytest.mark.parametrize("device, dtype", test_params)
def test_factories_and_properties(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m_modes = [2, 3, 4]
    n_modes = [4, 3, 2]

    # Test zeros factory
    tt_z = Operator(
        TTEngine.zeros(m_modes, n_modes, device=torch.device(device), dtype=dtype)
    )
    assert isinstance(tt_z, Operator)
    assert tt_z.is_tt
    assert tt_z.as_tt().m_modes == m_modes
    assert tt_z.as_tt().n_modes == n_modes
    assert (
        tt_z.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert tt_z.dtype == dtype

    for core in tt_z.as_tt().cores:
        torch.testing.assert_close(core, torch.zeros_like(core))

    # Test ones factory
    tt_o = Operator(
        TTEngine.ones(m_modes, n_modes, device=torch.device(device), dtype=dtype)
    )
    assert isinstance(tt_o, Operator)
    assert tt_o.as_tt().m_modes == m_modes
    assert tt_o.as_tt().n_modes == n_modes

    for core in tt_o.as_tt().cores:
        torch.testing.assert_close(core, torch.ones_like(core))

    # Check other properties
    assert len(tt_z.as_tt().ranks) == len(m_modes) + 1
    assert len(tt_z.as_tt().free_indices) > 0


@pytest.mark.parametrize("device, dtype", test_params)
def test_math_operators(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m_modes = [2, 3]
    n_modes = [2, 3]

    # Create operators and tensors for math operations
    tt1 = Operator(
        TTEngine.ones(m_modes, n_modes, device=torch.device(device), dtype=dtype)
    )
    tt2 = Operator(
        TTEngine.ones(m_modes, n_modes, device=torch.device(device), dtype=dtype)
    )
    scalar = 2.0
    tensor_scalar = torch.tensor(3.0, device=device, dtype=dtype)
    assert tt1.is_tt and tt2.is_tt

    # Helper to generate dynamically sized expected dense outputs
    def expected_dense(val, reference_tt):
        ref_dense = reference_tt.as_tt().to_dense()
        return val * torch.ones_like(ref_dense)

    # 1. Operator-to-Operator operations
    # Addition
    tt_add = tt1 + tt2
    assert isinstance(tt_add, Operator)
    torch.testing.assert_close(tt_add.as_tt().to_dense(), expected_dense(2.0, tt1))

    # Subtraction
    tt_sub = tt1 - tt2
    assert isinstance(tt_sub, Operator)
    torch.testing.assert_close(tt_sub.as_tt().to_dense(), expected_dense(0.0, tt1))

    # Element-wise multiplication / division
    tt_mul = tt1 * tt2
    assert isinstance(tt_mul, Operator)
    torch.testing.assert_close(tt_mul.as_tt().to_dense(), expected_dense(1.0, tt1))

    tt_div = tt1 / tt2
    assert isinstance(tt_div, Operator)
    torch.testing.assert_close(tt_div.as_tt().to_dense(), expected_dense(1.0, tt1))

    # 2. Scalar operations
    assert isinstance(tt1 + scalar, Operator)
    torch.testing.assert_close(
        (tt1 + scalar).as_tt().to_dense(), expected_dense(3.0, tt1)
    )

    assert isinstance(scalar + tt1, Operator)
    torch.testing.assert_close(
        (scalar + tt1).as_tt().to_dense(), expected_dense(3.0, tt1)
    )

    assert isinstance(tt1 - scalar, Operator)
    torch.testing.assert_close(
        (tt1 - scalar).as_tt().to_dense(), expected_dense(-1.0, tt1)
    )

    assert isinstance(scalar - tt1, Operator)
    torch.testing.assert_close(
        (scalar - tt1).as_tt().to_dense(), expected_dense(1.0, tt1)
    )

    assert isinstance(tt1 * scalar, Operator)
    torch.testing.assert_close(
        (tt1 * scalar).as_tt().to_dense(), expected_dense(2.0, tt1)
    )

    assert isinstance(scalar * tt1, Operator)
    torch.testing.assert_close(
        (scalar * tt1).as_tt().to_dense(), expected_dense(2.0, tt1)
    )

    assert isinstance(tt1 / scalar, Operator)
    torch.testing.assert_close(
        (tt1 / scalar).as_tt().to_dense(), expected_dense(0.5, tt1)
    )

    assert isinstance(scalar / tt1, Operator)
    torch.testing.assert_close(
        (scalar / tt1).as_tt().to_dense(), expected_dense(2.0, tt1)
    )

    # 3. Tensor operations
    assert isinstance(tt1 + tensor_scalar, Operator)
    torch.testing.assert_close(
        (tt1 + tensor_scalar).as_tt().to_dense(), expected_dense(4.0, tt1)
    )

    assert isinstance(tt1 * tensor_scalar, Operator)
    torch.testing.assert_close(
        (tt1 * tensor_scalar).as_tt().to_dense(), expected_dense(3.0, tt1)
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_inplace_negation(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tt = Operator(
        TTEngine.ones([2, 2], [2, 2], device=torch.device(device), dtype=dtype)
    )
    assert tt.is_tt

    # Apply in-place negation
    tt.neg_()

    # Check only the first core is negated
    torch.testing.assert_close(
        tt.as_tt().cores[0], -torch.ones_like(tt.as_tt().cores[0])
    )

    for core in tt.as_tt().cores[1:]:
        # Check that the remaining cores are completely positive ones
        torch.testing.assert_close(core, torch.ones_like(core))


@pytest.mark.parametrize("device, dtype", test_params)
def test_orthogonalize_and_round(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tt = Operator(
        TTEngine.ones([4, 4, 4], [4, 4, 4], device=torch.device(device), dtype=dtype)
    )
    assert tt.is_tt

    # Out-of-place orthogonalization
    tt_ortho = Operator(tt.as_tt().lr_orthogonalize())
    assert isinstance(tt_ortho, Operator)
    assert tt_ortho is not tt

    # In-place orthogonalization
    tt.as_tt().lr_orthogonalize_()

    # Out-of-place rounding
    tt_rounded = tt.round(eps=1e-3, max_rank=2)
    assert isinstance(tt_rounded, Operator)

    # Ranks should ideally be reduced
    assert all(r <= 2 for r in tt_rounded.as_tt().ranks[1:-1])

    # In-place rounding
    tt.round_(eps=1e-3, max_rank=2)
    assert all(r <= 2 for r in tt.as_tt().ranks[1:-1])


@pytest.mark.parametrize("device, dtype", test_params)
def test_from_dense_to_dense(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m_modes = [2, 3]
    n_modes = [3, 2]

    # Create a dense tensor with shape (m_0, m_1, n_0, n_1) for default interleaved=False
    dense_tensor = torch.randn(*m_modes, *n_modes, device=device, dtype=dtype)

    # Decompose into TT-Operator
    tt = Operator(
        TTEngine.from_dense(
            dense_tensor,
            m_modes=m_modes,
            n_modes=n_modes,
            eps=1e-10,
            is_interleaved=False,
        )
    )

    assert tt.is_tt
    assert isinstance(tt, Operator)
    assert tt.as_tt().m_modes == m_modes
    assert tt.as_tt().n_modes == n_modes

    # Reconstruct back to dense
    dense_reconstructed = tt.as_tt().to_dense(interleave=False)

    # The reconstructed tensor should match the original within the specified truncation tolerance
    torch.testing.assert_close(dense_tensor, dense_reconstructed, rtol=1e-3, atol=1e-4)
