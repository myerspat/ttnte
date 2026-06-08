import pytest
import torch

from ttnte.linalg import State, TTEngine

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

    tt = State(TTEngine(cores), name)

    assert tt.is_tt
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
            tt.as_tt().cores[i - 1],
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

    tt0 = State(TTEngine(cores))

    # Send to GPU as dtype
    tt1 = tt0.to(torch.device("cuda", 0), dtype)

    # Check the getters
    assert tt0.is_tt and tt1.is_tt
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

    # Test zeros factory
    tt_z = State(TTEngine.zeros(m_modes, device=torch.device(device), dtype=dtype))
    assert tt_z.is_tt
    assert isinstance(tt_z, State)
    assert tt_z.as_tt().m_modes == m_modes
    assert (
        tt_z.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert tt_z.dtype == dtype

    for core in tt_z.as_tt().cores:
        torch.testing.assert_close(core, torch.zeros_like(core))

    # Test ones factory
    tt_o = State(TTEngine.ones(m_modes, device=torch.device(device), dtype=dtype))
    assert tt_o.is_tt
    assert isinstance(tt_o, State)
    assert tt_o.as_tt().m_modes == m_modes

    for core in tt_o.as_tt().cores:
        torch.testing.assert_close(core, torch.ones_like(core))

    # Check other properties
    assert len(tt_z.as_tt().ranks) == len(m_modes) + 1
    assert len(tt_z.as_tt().free_indices) > 0


@pytest.mark.parametrize("device, dtype", test_params)
def test_math_operators(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create states and tensors for math operations
    tt1 = State(TTEngine.ones([2, 3], device=torch.device(device), dtype=dtype))
    tt2 = State(TTEngine.ones([2, 3], device=torch.device(device), dtype=dtype))
    scalar = 2.0
    tensor_scalar = torch.tensor(3.0, device=device, dtype=dtype)
    assert tt1.is_tt and tt2.is_tt

    # 1. State-to-State operations
    # Addition
    tt_add = tt1 + tt2
    assert isinstance(tt_add, State)
    torch.testing.assert_close(
        tt_add.as_tt().to_dense().squeeze(),
        2 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )

    # Subtraction (Note: C++ header has a typo here `a + b` instead of `a - b`
    # for the State-TTState operator, but we test the interface regardless)
    tt_sub = tt1 - tt2
    assert isinstance(tt_sub, State)
    torch.testing.assert_close(
        tt_sub.as_tt().to_dense().squeeze(),
        torch.zeros([2, 3], device=torch.device(device), dtype=dtype),
    )

    # Element-wise multiplication / division
    tt_mul = tt1 * tt2
    assert isinstance(tt_mul, State)
    torch.testing.assert_close(
        tt_mul.as_tt().to_dense().squeeze(),
        torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )

    tt_div = tt1 / tt2
    assert isinstance(tt_div, State)
    torch.testing.assert_close(
        tt_div.as_tt().to_dense().squeeze(),
        torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )

    # 2. Scalar operations
    assert isinstance(tt1 + scalar, State)
    torch.testing.assert_close(
        (tt1 + scalar).as_tt().to_dense().squeeze(),
        3 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(scalar + tt1, State)
    torch.testing.assert_close(
        (scalar + tt1).as_tt().to_dense().squeeze(),
        3 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(tt1 - scalar, State)
    torch.testing.assert_close(
        (tt1 - scalar).as_tt().to_dense().squeeze(),
        -1 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(scalar - tt1, State)
    torch.testing.assert_close(
        (scalar - tt1).as_tt().to_dense().squeeze(),
        torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(tt1 * scalar, State)
    torch.testing.assert_close(
        (tt1 * scalar).as_tt().to_dense().squeeze(),
        2 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(scalar * tt1, State)
    torch.testing.assert_close(
        (scalar * tt1).as_tt().to_dense().squeeze(),
        2 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(tt1 / scalar, State)
    torch.testing.assert_close(
        (tt1 / scalar).as_tt().to_dense().squeeze(),
        0.5 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(scalar / tt1, State)
    torch.testing.assert_close(
        (scalar / tt1).as_tt().to_dense().squeeze(),
        2 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )

    # 3. Tensor operations
    assert isinstance(tt1 + tensor_scalar, State)
    torch.testing.assert_close(
        (tt1 + tensor_scalar).as_tt().to_dense().squeeze(),
        4 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )
    assert isinstance(tt1 * tensor_scalar, State)
    torch.testing.assert_close(
        (tt1 * tensor_scalar).as_tt().to_dense().squeeze(),
        3 * torch.ones([2, 3], device=torch.device(device), dtype=dtype),
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_inplace_negation(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tt = State(TTEngine.ones([2, 2], device=torch.device(device), dtype=dtype))
    assert tt.is_tt

    # Apply in-place negation
    tt.neg_()

    # Check only the first core is negated
    torch.testing.assert_close(
        tt.as_tt().cores[0], -torch.ones_like(tt.as_tt().cores[0])
    )

    for core in tt.as_tt().cores[1:]:
        # Check that the core is completely negative ones
        torch.testing.assert_close(core, torch.ones_like(core))


@pytest.mark.parametrize("device, dtype", test_params)
def test_orthogonalize_and_round(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tt = State(TTEngine.ones([4, 4, 4], device=torch.device(device), dtype=dtype))
    assert tt.is_tt

    # Out-of-place orthogonalization
    tt_ortho = State(tt.as_tt().lr_orthogonalize())
    assert tt_ortho.is_tt
    assert isinstance(tt_ortho, State)
    assert tt_ortho is not tt

    # In-place orthogonalization
    tt.as_tt().lr_orthogonalize_()

    # Out-of-place rounding
    tt_rounded = tt.round(eps=1e-3, max_rank=2)
    assert isinstance(tt_rounded, State)

    # Ranks should ideally be reduced
    assert all(r <= 2 for r in tt_rounded.as_tt().ranks[1:-1])

    # In-place rounding
    tt.round_(eps=1e-3, max_rank=2)
    assert all(r <= 2 for r in tt.as_tt().ranks[1:-1])


@pytest.mark.parametrize("device, dtype", test_params)
def test_pack_unpack_roundtrip(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m_modes = [2, 3, 4]
    state = State(TTEngine.ones(m_modes, device=torch.device(device), dtype=dtype))

    buf = state.pack()

    assert buf.dim() == 1
    assert buf.device == torch.device("cpu")
    assert buf.dtype == dtype

    unpacked = State.unpack(buf)

    assert unpacked.is_tt
    assert unpacked.as_tt().m_modes == m_modes
    assert unpacked.as_tt().ranks == state.as_tt().ranks
    assert unpacked.as_tt().dtype == dtype

    for orig, restored in zip(state.as_tt().cores, unpacked.as_tt().cores):
        torch.testing.assert_close(orig.cpu(), restored)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pack_preallocated_buffer(dtype):
    state = State(TTEngine.ones([3, 4], dtype=dtype))
    ref_buf = state.pack()
    pre_buf = torch.empty_like(ref_buf)

    returned = state.pack(pre_buf)

    assert returned.data_ptr() == pre_buf.data_ptr()
    torch.testing.assert_close(returned, ref_buf)


@pytest.mark.parametrize(
    "src_dtype, dst_dtype",
    [
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
    ],
)
def test_pack_cross_precision(src_dtype, dst_dtype):
    state = State(TTEngine.ones([2, 3], dtype=src_dtype))
    ref_buf = state.pack()
    cross_buf = torch.empty(ref_buf.numel(), dtype=dst_dtype)
    state.pack(cross_buf)

    unpacked = State.unpack(cross_buf)
    m_modes = unpacked.as_tt().m_modes
    # Strip trailing n-mode dims (all 1 for a vector TT)
    dense = unpacked.as_tt().to_dense().reshape(m_modes)
    torch.testing.assert_close(dense, torch.ones_like(dense), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_lower_face(dtype):
    m_modes = [3, 4, 5]
    dense = torch.arange(3 * 4 * 5, dtype=dtype).reshape(m_modes)
    state = State(TTEngine.from_dense(dense, eps=0.0))
    # Use to_dense() of the full TT as reference to avoid from_dense roundoff
    full_dense = state.as_tt().to_dense().reshape(m_modes)

    for dim in range(len(m_modes)):
        narrowed = state.narrow(dim, 0, 1)
        assert narrowed.as_tt().m_modes[dim] == 1
        # to_dense() returns [m0,...,mK-1, 1,...,1]; reshape to remove n-mode dims
        narrowed_dense = narrowed.as_tt().to_dense().reshape(narrowed.as_tt().m_modes)
        torch.testing.assert_close(narrowed_dense, full_dense.narrow(dim, 0, 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_upper_face(dtype):
    m_modes = [3, 4, 5]
    dense = torch.arange(3 * 4 * 5, dtype=dtype).reshape(m_modes)
    state = State(TTEngine.from_dense(dense, eps=0.0))
    full_dense = state.as_tt().to_dense().reshape(m_modes)

    for dim, m in enumerate(m_modes):
        narrowed = state.narrow(dim, -1, 1)
        assert narrowed.as_tt().m_modes[dim] == 1
        narrowed_dense = narrowed.as_tt().to_dense().reshape(narrowed.as_tt().m_modes)
        torch.testing.assert_close(narrowed_dense, full_dense.narrow(dim, m - 1, 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_inplace(dtype):
    m_modes = [3, 4, 5]
    dense = torch.arange(3 * 4 * 5, dtype=dtype).reshape(m_modes)
    state = State(TTEngine.from_dense(dense, eps=0.0))

    ref = state.narrow(1, -1, 1).as_tt().to_dense()
    state.narrow_(1, -1, 1)

    assert state.as_tt().m_modes[1] == 1
    torch.testing.assert_close(state.as_tt().to_dense(), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_does_not_mutate(dtype):
    m_modes = [3, 4, 5]
    state = State(TTEngine.ones(m_modes, dtype=dtype))
    original_modes = list(state.as_tt().m_modes)

    _ = state.narrow(1, 0, 1)

    assert list(state.as_tt().m_modes) == original_modes
