import pytest
import torch

from ttnte.linalg import TTEngine

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]


# =============================================================================
# pack / unpack
# =============================================================================


@pytest.mark.parametrize("device, dtype", test_params)
def test_pack_unpack_roundtrip(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    engine = TTEngine.ones([2, 3, 4], device=torch.device(device), dtype=dtype)

    buf = engine.pack()

    assert buf.dim() == 1
    assert buf.device == torch.device("cpu")
    assert buf.dtype == dtype

    restored = TTEngine.unpack(buf)

    assert restored.m_modes == engine.m_modes
    assert restored.n_modes == engine.n_modes
    assert restored.ranks == engine.ranks
    assert restored.dtype == dtype

    for orig, rest in zip(engine.cores, restored.cores):
        torch.testing.assert_close(orig.cpu(), rest)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pack_preallocated_buffer(dtype):
    engine = TTEngine.ones([3, 4], dtype=dtype)
    ref_buf = engine.pack()
    pre_buf = torch.empty_like(ref_buf)

    returned = engine.pack(pre_buf)

    assert returned.data_ptr() == pre_buf.data_ptr()
    torch.testing.assert_close(returned, ref_buf)


# =============================================================================
# narrow — m-modes (uninterleaved, interleaved)
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_m_mode_uninterleaved(dtype):
    """Narrow(dim, ..., interleaved=False) for dim < K narrows the m-mode."""
    m_modes = [3, 4, 5]
    dense = torch.arange(3 * 4 * 5, dtype=dtype).reshape(m_modes)
    engine = TTEngine.from_dense(dense, eps=0.0)
    # Use to_dense() of the full TT as reference to avoid from_dense roundoff
    dense_full = engine.to_dense().reshape(m_modes)

    for dim in range(len(m_modes)):
        narrowed = engine.narrow(dim, 0, 1)
        assert narrowed.m_modes[dim] == 1
        narrowed_dense = narrowed.to_dense().reshape(narrowed.m_modes)
        torch.testing.assert_close(narrowed_dense, dense_full.narrow(dim, 0, 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_m_mode_interleaved(dtype):
    """Narrow(dim, ..., interleaved=True) for even dim narrows the m-mode."""
    m_modes = [3, 4, 5]
    dense = torch.arange(3 * 4 * 5, dtype=dtype).reshape(m_modes)
    engine = TTEngine.from_dense(dense, eps=0.0)
    dense_full = engine.to_dense().reshape(m_modes)

    for dim in range(len(m_modes)):
        # interleaved even dim 2*d corresponds to m-mode of core d
        narrowed = engine.narrow(2 * dim, 0, 1, interleaved=True)
        assert narrowed.m_modes[dim] == 1
        narrowed_dense = narrowed.to_dense().reshape(narrowed.m_modes)
        torch.testing.assert_close(narrowed_dense, dense_full.narrow(dim, 0, 1))


# =============================================================================
# narrow — n-modes (requires an operator TTEngine with n_modes > 1)
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_n_mode_uninterleaved(dtype):
    """Narrow(K+dim, ..., interleaved=False) narrows the n-mode of core dim."""
    m_modes = [3, 4]
    n_modes = [5, 6]
    engine = TTEngine.ones(m_modes, n_modes, dtype=dtype)
    K = len(m_modes)
    # to_dense(interleave=False) → [m0, m1, n0, n1]
    dense_full = engine.to_dense(interleave=False)

    for d in range(K):
        narrowed = engine.narrow(K + d, 0, 1, interleaved=False)
        assert narrowed.n_modes[d] == 1
        assert narrowed.m_modes[d] == m_modes[d]
        # to_dense after narrowing n-mode d: shape changes at dim K+d
        narrowed_dense = narrowed.to_dense(interleave=False)
        torch.testing.assert_close(narrowed_dense, dense_full.narrow(K + d, 0, 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_n_mode_interleaved(dtype):
    """Narrow(2*dim+1, ..., interleaved=True) narrows the n-mode of core dim."""
    m_modes = [3, 4]
    n_modes = [5, 6]
    engine = TTEngine.ones(m_modes, n_modes, dtype=dtype)
    # to_dense(interleave=True) → [m0, n0, m1, n1]
    dense_full = engine.to_dense(interleave=True)

    for d in range(len(m_modes)):
        narrowed = engine.narrow(2 * d + 1, 0, 1, interleaved=True)
        assert narrowed.n_modes[d] == 1
        assert narrowed.m_modes[d] == m_modes[d]
        # interleaved dim 2*d+1 → to_dense(interleave=True) dim 2*d+1
        narrowed_dense = narrowed.to_dense(interleave=True)
        torch.testing.assert_close(narrowed_dense, dense_full.narrow(2 * d + 1, 0, 1))


# =============================================================================
# narrow — in-place and mutation safety
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_inplace(dtype):
    engine = TTEngine.ones([3, 4, 5], dtype=dtype)
    ref = engine.narrow(1, -1, 1).to_dense()

    engine.narrow_(1, -1, 1)

    assert engine.m_modes[1] == 1
    torch.testing.assert_close(engine.to_dense(), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_narrow_does_not_mutate(dtype):
    engine = TTEngine.ones([3, 4, 5], dtype=dtype)
    original_m = list(engine.m_modes)
    original_n = list(engine.n_modes)

    _ = engine.narrow(1, 0, 1)

    assert list(engine.m_modes) == original_m
    assert list(engine.n_modes) == original_n
