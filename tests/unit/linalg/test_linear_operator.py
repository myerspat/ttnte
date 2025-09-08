import torch as tn
import numpy as np
from torchtt import eye
import pytest

from ttnte.linalg import TTOperator as TTO_python
from ttnte.linalg import SparseOperator as SO_python

try:
    from ttnte.cpp.linalg import TTOperator as TTO_cpp
    from ttnte.cpp.linalg import SparseOperator as SO_cpp
    from ttnte.cpp.linalg import LinearOperator as LO_cpp

    cpp_available = True

except:
    cpp_available = False


def run_linear_operator(SparseOperator, TTOperator, LinearOperator):
    # Shape of input
    shape = [5, 10, 15]

    # Create an identity operator
    I_tntt = eye(shape)
    I_tn = tn.eye(np.prod(shape))

    # Pass to TTOperator
    I = TTOperator(I_tntt) + SparseOperator(I_tn)

    # Test matmul method with arbitrary vector
    a = tn.rand(shape)
    b = I @ a
    assert tn.equal(0 * a, b)


@pytest.mark.skipif(not cpp_available, reason="C++ backend failed to import")
def test_linear_operator_cpp():
    run_linear_operator(SO_cpp, TTO_cpp, LO_cpp)
