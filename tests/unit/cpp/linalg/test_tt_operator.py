import torch as tn
from torchtt import eye

from ttnte.linalg import TTOperator


def test_tt_operator():
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

    # Check each core
    for i in range(I.num_cores):
        if i == 0:
            assert tn.equal(I.cores[i], I_tntt.cores[i].squeeze(0))
        elif i == I.num_cores - 1:
            assert tn.equal(I.cores[i], I_tntt.cores[i].squeeze(3))
        else:
            assert tn.equal(I.cores[i], I_tntt.cores[i])

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
