import torch as tn
import numpy as np
from torchtt import eye
import pytest

from ttnte.linalg.tt_operator import TTOperator as TTO_python
from ttnte.linalg.sparse_operator import SparseOperator as SO_python

try:
    from ttnte.cpp.linalg import TTOperator as TTO_cpp
    from ttnte.cpp.linalg import SparseOperator as SO_cpp

    cpp_available = True

except:
    cpp_available = False


def run_linear_operator(SparseOperator, TTOperator):
    # Shape of input
    shape = [5, 10, 15]

    # Create an identity operator
    I_tntt = eye(shape)
    I_tn = tn.eye(np.prod(shape))

    # Pass to TTOperator
    I = (
        TTOperator(I_tntt.clone())
        + SparseOperator(I_tn.clone())
        - TTOperator(I_tntt.clone())
        + (-SparseOperator(I_tn.clone()))
    )

    # Check there are 4 operators
    assert len(I.operators) == 4

    # Test matmul method with arbitrary vector
    a = tn.rand(shape)
    b = I @ a
    assert tn.equal(tn.zeros(b.shape), b)

    # Combine TTOperators and SparseOperators
    I = I.combine()

    # Check there are 2 operators
    assert len(I.operators) == 2

    # Test matmul method with arbitrary vector
    a = tn.rand(shape)
    b = I @ a
    assert tn.equal(tn.zeros(b.shape), b)

    # ================================================
    # Test rounding
    I = (
        TTOperator(I_tntt.clone())
        + SparseOperator(I_tn.clone())
        - TTOperator(I_tntt.clone())
        + (-SparseOperator(I_tn.clone()))
    )

    # Combine TTOperators and SparseOperators
    I = I.round(1e-10)

    # Check there are 2 operators
    assert len(I.operators) == 3

    # Test matmul method with arbitrary vector
    a = tn.rand(shape)
    b = I @ a
    tn.testing.assert_close(tn.zeros(b.shape), b)


def test_linear_operator_python():
    run_linear_operator(SO_python, TTO_python)


@pytest.mark.skipif(not cpp_available, reason="C++ backend failed to import")
def test_linear_operator_cpp():
    run_linear_operator(SO_cpp, TTO_cpp)
