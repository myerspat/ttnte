import pytest
import torch
import math
from ttnte.linalg import (
    TTEngine,
    elementwise_divide,
    dmrg_mv,
    fast_hadamard,
    fast_mm,
    fast_mv,
    amen_mm,
    amen_mv,
    amen_solve,
)

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]

# Set a seed so iterative solvers don't randomly fail due to bad initializations
torch.manual_seed(42)

# --- Helper Functions for Dense Math ---


def dense_op_to_matrix(tensor, m_modes, n_modes):
    """Flattens a (m_0, m_1, n_0, n_1) tensor into a (M, N) matrix."""
    M = math.prod(m_modes)
    N = math.prod(n_modes)
    return tensor.reshape(M, N)


def matrix_to_dense_op(matrix, m_modes, n_modes):
    """Reshapes a (M, N) matrix back to (m_0, m_1, n_0, n_1)."""
    return matrix.reshape(*m_modes, *n_modes)


# ---------------------------------------


@pytest.mark.parametrize("device, dtype", test_params)
def test_elementwise_operations(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    modes = [4, 5]

    # Use random data, but keep denominator safely away from zero
    dense_a = torch.randn(*modes, device=device, dtype=dtype)
    dense_b = torch.rand(*modes, device=device, dtype=dtype) + 1.0

    tt_a = TTEngine.from_dense(dense_a, eps=1e-12)
    tt_b = TTEngine.from_dense(dense_b, eps=1e-12)

    # 1. Hadamard (Element-wise multiplication)
    tt_hadamard = fast_hadamard(tt_a, tt_b, eps=1e-8)
    dense_expected_hadamard = dense_a * dense_b
    torch.testing.assert_close(
        tt_hadamard.to_dense().squeeze(), dense_expected_hadamard, rtol=1e-3, atol=1e-4
    )

    # 2. Divide (Element-wise division)
    tt_div = elementwise_divide(tt_a, tt_b, eps=1e-8)
    dense_expected_div = dense_a / dense_b
    torch.testing.assert_close(
        tt_div.to_dense().squeeze(), dense_expected_div, rtol=1e-3, atol=1e-4
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_matrix_vector_operations(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m_modes = [3, 4]
    n_modes = [4, 3]

    dense_A = torch.randn(*m_modes, *n_modes, device=device, dtype=dtype)
    dense_x = torch.randn(*n_modes, device=device, dtype=dtype)

    # Convert to TT
    tt_A = TTEngine.from_dense(
        dense_A, m_modes=m_modes, n_modes=n_modes, is_interleaved=False, eps=1e-12
    )
    tt_x = TTEngine.from_dense(dense_x, eps=1e-12)

    # Compute Ground Truth Dense Math: y = A @ x
    A_mat = dense_op_to_matrix(dense_A, m_modes, n_modes)
    x_vec = dense_x.flatten()
    y_expected = (A_mat @ x_vec).reshape(*m_modes)

    # 1. Fast MV
    y_fast = fast_mv(tt_A, tt_x, eps=1e-8)
    torch.testing.assert_close(
        y_fast.to_dense().squeeze(), y_expected, rtol=1e-3, atol=1e-4
    )

    # 2. DMRG MV
    y_dmrg = dmrg_mv(tt_A, tt_x, eps=1e-8)
    torch.testing.assert_close(
        y_dmrg.to_dense().squeeze(), y_expected, rtol=1e-3, atol=1e-4
    )

    # 3. AMEn MV
    y_amen = amen_mv(tt_A, tt_x, eps=1e-8)
    torch.testing.assert_close(
        y_amen.to_dense().squeeze(), y_expected, rtol=1e-3, atol=1e-4
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_matrix_matrix_operations(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m_modes_A = [3, 4]
    n_modes_A = [2, 3]  # This is m_modes_B
    n_modes_B = [4, 2]

    dense_A = torch.randn(*m_modes_A, *n_modes_A, device=device, dtype=dtype)
    dense_B = torch.randn(*n_modes_A, *n_modes_B, device=device, dtype=dtype)

    tt_A = TTEngine.from_dense(
        dense_A, m_modes=m_modes_A, n_modes=n_modes_A, is_interleaved=False, eps=1e-12
    )
    tt_B = TTEngine.from_dense(
        dense_B, m_modes=n_modes_A, n_modes=n_modes_B, is_interleaved=False, eps=1e-12
    )

    # Compute Ground Truth Dense Math: C = A @ B
    A_mat = dense_op_to_matrix(dense_A, m_modes_A, n_modes_A)
    B_mat = dense_op_to_matrix(dense_B, n_modes_A, n_modes_B)
    C_expected_mat = A_mat @ B_mat
    C_expected = matrix_to_dense_op(C_expected_mat, m_modes_A, n_modes_B)

    # 1. Fast MM
    C_fast = fast_mm(tt_A, tt_B, eps=1e-8)
    torch.testing.assert_close(
        C_fast.to_dense(interleave=False).squeeze(), C_expected, rtol=1e-3, atol=1e-4
    )

    # 2. AMEn MM
    C_amen = amen_mm(tt_A, tt_B, eps=1e-8)
    torch.testing.assert_close(
        C_amen.to_dense(interleave=False).squeeze(), C_expected, rtol=1e-3, atol=1e-4
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_amen_solve(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    modes = [4, 4]
    N = math.prod(modes)

    # To ensure the iterative AMEn solver actually converges,
    # we MUST generate a Symmetric Positive Definite (SPD) matrix.
    X_mat = torch.randn(N, N, device=device, dtype=dtype)
    A_mat = X_mat @ X_mat.T + torch.eye(N, device=device, dtype=dtype) * 5.0

    dense_A = matrix_to_dense_op(A_mat, modes, modes)
    dense_b = torch.randn(*modes, device=device, dtype=dtype)

    tt_A = TTEngine.from_dense(
        dense_A, m_modes=modes, n_modes=modes, is_interleaved=False, eps=1e-12
    )
    tt_b = TTEngine.from_dense(dense_b, eps=1e-12)

    # Compute Ground Truth Dense Math: x = A^-1 b
    b_vec = dense_b.flatten()
    x_expected_vec = torch.linalg.solve(A_mat, b_vec)
    x_expected = x_expected_vec.reshape(*modes)

    # AMEn Solve
    tt_x = amen_solve(tt_A, tt_b, eps=1e-8)

    # Iterative solvers can be slightly less precise, so tolerances are relaxed here
    torch.testing.assert_close(
        tt_x.to_dense().squeeze(), x_expected, rtol=1e-2, atol=1e-3
    )
