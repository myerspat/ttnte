import torch as tn
import numpy as np
from torchtt import eye, random
import cotengra as ctg
import pytest

from ttnte.linalg.tt_operator import TTOperator as TTO_python

try:
    from ttnte.cpp.linalg import TTOperator as TTO_cpp

    cpp_available = True

except ImportError:
    cpp_available = False


def run_tt_operator(TTOperator):
    tn.set_default_dtype(tn.float64)

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
    assert I.dtype == tn.float64
    assert I.device == tn.device("cpu")
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

    # Test type casting
    I = I.type(tn.float32)
    assert I.dtype == tn.float32

    # ==================================================================
    # Get random tensor train
    tt_rand_tntt = random([(10, 10), (15, 15), (20, 20)], R=10)
    tt_rand_tntt += tt_rand_tntt

    # Create TTOperator
    tt_rand = TTOperator(tt_rand_tntt.clone())

    # Test matmul for this random tensor using torchtt as the expected
    a = tn.rand([10, 15, 20])
    a_expected = (
        ctg.einsum("abcd,defg,ghij,cfi->abehj", *tt_rand_tntt.cores, a)
        .squeeze_(0)
        .squeeze_(-1)
    )
    a_ttop = tt_rand @ a

    # Check tensors are close
    tn.testing.assert_close(a_ttop, a_expected)

    # ==================================================================
    # Test rounding
    expected = (tt_rand_tntt).round(1e-10)
    actual = tt_rand.round(1e-10)

    # Check the tt was copied
    for core_a, core_e in zip(actual.cores, tt_rand.cores):
        assert not tn.equal(core_a, core_e)

    # Checks
    assert expected.R[1:-1] == actual.ranks
    assert expected.shape == actual.shape

    a = tn.rand([10, 15, 20])
    a_expected = (
        ctg.einsum("abcd,defg,ghij,cfi->abehj", *tt_rand_tntt.cores, a)
        .squeeze_(0)
        .squeeze_(-1)
    )
    a_ttop = tt_rand @ a
    # Check tensors are close
    tn.testing.assert_close(a_ttop, a_expected)


def test_tt_operator_python():
    run_tt_operator(TTO_python)


@pytest.mark.skipif(not cpp_available, reason="C++ backend failed to import")
def test_tt_operator_cpp():
    run_tt_operator(TTO_cpp)
