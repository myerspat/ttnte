import torch as tn
import numpy as np
from torchtt import eye
import pytest

from ttnte.linalg.tt_operator import TTOperator as TTO_python

try:
    from ttnte.cpp.linalg import TTOperator as TTO_cpp

    cpp_available = True

except ImportError:
    cpp_available = False


def run_tt_operator(TTOperator):
    # Shape of input
    shape = [10, 15, 20, 25, 30]

    # Create an identity operator
    I_tntt = eye(shape)

    # Pass to TTOperator
    I = TTOperator(I_tntt)

    # Run checks
    assert I.num_cores == 5
    assert I.input_shape == list(shape)
    assert I.output_shape == list(shape)
    assert I.shape == [(s, s) for s in shape]
    assert I.nelements == np.sum([s * s for s in shape])
    assert I.compression == (np.prod(shape) ** 2) / I.nelements

    # Check each core
    for i in range(I.num_cores):
        assert tn.equal(I.cores[i], I_tntt.cores[i].reshape(I.cores[i].shape))

    # Check pass to cuda
    if tn.cuda.is_available() and tn.cuda.device_count() > 0:
        # Pass to GPU
        I.cuda(0)

        # Iterate through cores
        for core in I.cores:
            assert core.get_device() == 0

        # Take off GPU
        I.cpu()

        for core in I.cores:
            assert core.get_device() == -1

    # Test matmul method with arbitrary vector
    a = tn.rand(shape)
    b = I @ a
    assert tn.equal(a, b)


def test_tt_operator_python():
    run_tt_operator(TTO_python)


@pytest.mark.skipif(not cpp_available, reason="C++ backend failed to import")
def test_tt_operator_cpp():
    run_tt_operator(TTO_cpp)
