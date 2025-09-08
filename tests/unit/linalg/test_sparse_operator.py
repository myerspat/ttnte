import torch as tn
import pytest

from ttnte.linalg.sparse_operator import SparseOperator as SO_python

try:
    from ttnte.cpp.linalg import SparseOperator as SO_cpp

    cpp_available = True

except ImportError:
    cpp_available = False


def run_sparse_operator(SparseOperator):
    n = 500

    # Create an identity operator
    I_tn = tn.eye(n).to_sparse_csr()

    # Pass to TTOperator
    I = SparseOperator(I_tn)

    # Run checks
    assert I.input_shape == [n]
    assert I.output_shape == [n]
    assert I.shape == (n, n)
    assert I.nnz == n
    assert I.nelements == 3 * n + 1
    assert I.compression == n * n / (3 * n + 1)
    assert tn.equal(I.tensor.values(), I_tn.values())
    assert tn.equal(I.tensor.crow_indices(), I_tn.crow_indices())
    assert tn.equal(I.tensor.col_indices(), I_tn.col_indices())

    # Check pass to cuda
    if tn.cuda.is_available() and tn.cuda.device_count() > 0:
        # Pass to GPU
        I.cuda(0)
        assert I.tensor.get_device() == 0

        # Take off GPU
        I.cpu()
        assert I.tensor.get_device() == -1

    # Test matmul method with arbitrary vector
    a = tn.rand(n)
    b = I @ a
    assert tn.equal(a, b)


def test_tt_operator_python():
    run_sparse_operator(SO_python)


@pytest.mark.skipif(not cpp_available, reason="C++ backend failed to import")
def test_csr_operator_cpp():
    run_sparse_operator(SO_cpp)
