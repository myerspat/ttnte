import torch
import pytest

from ttnte.linalg import State, Operator, LinearSystem, TTEngine

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
    operator = Operator(TTEngine(cores), "operator")

    cores = [
        2 * torch.ones((i * 10), device=device, dtype=dtype).reshape((1, i * 10, 1))
        for i in range(1, 4)
    ]
    state = State(TTEngine.clone_from(cores), "state")
    source = State(TTEngine.clone_from(cores), "source")
    assert operator.is_tt and state.is_tt and source.is_tt

    assert (
        operator.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert operator.dtype == dtype
    assert (
        state.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert state.dtype == dtype
    assert (
        source.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert source.dtype == dtype

    # Create a linear system
    ls = LinearSystem(operator, state, source)

    new_operator = ls.interior_op
    new_state = ls.state
    new_source = ls.source

    assert (
        new_operator.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert new_operator.dtype == dtype
    assert (
        new_state.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert new_state.dtype == dtype
    assert (
        new_source.device == torch.device(device)
        if device == "cpu"
        else torch.device(device, 0)
    )
    assert new_source.dtype == dtype

    for j, (op_core, st_core, so_core) in enumerate(
        zip(
            new_operator.as_tt().cores,
            new_state.as_tt().cores,
            new_source.as_tt().cores,
        )
    ):
        i = j + 1
        torch.testing.assert_close(
            op_core,
            torch.ones((i * 10, i * 10), device=device, dtype=dtype).reshape(
                (1, i * 10, i * 10, 1)
            ),
        )
        torch.testing.assert_close(
            st_core,
            2
            * torch.ones(i * 10, device=device, dtype=dtype).reshape((1, i * 10, 1, 1)),
        )
        torch.testing.assert_close(
            so_core,
            2
            * torch.ones(i * 10, device=device, dtype=dtype).reshape((1, i * 10, 1, 1)),
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
    operator = Operator(TTEngine(cores), "operator")

    cores = [
        2
        * torch.ones((i * 10), device="cpu", dtype=torch.float64).reshape(
            (1, i * 10, 1)
        )
        for i in range(1, 4)
    ]
    state = State(TTEngine.clone_from(cores), "state")
    source = State(TTEngine.clone_from(cores), "source")

    # Create linear system
    ls = LinearSystem(operator)
    ls.state = state
    ls.source = source

    # Test buffer send
    ls.transfer_buffer(torch.device("cuda", 0), dtype)

    torch.testing.assert_close(
        ls.get_buffer(torch.device("cpu")).to(torch.device("cuda", 0), dtype),
        ls.get_buffer(torch.device("cuda", 0)),
    )
    assert ls.interior_op.device == torch.device("cuda", 0)
    assert ls.interior_op.dtype == dtype
    assert ls.state.device == torch.device("cpu")
    assert ls.state.dtype == torch.float64
    assert ls.source.device == torch.device("cpu")
    assert ls.source.dtype == torch.float64

    ls.transfer_nonbuffer(torch.device("cuda", 0), dtype)
    assert ls.state.device == torch.device("cuda", 0)
    assert ls.state.dtype == dtype
    assert ls.source.device == torch.device("cuda", 0)
    assert ls.source.dtype == dtype

    ls.to_(torch.device("cpu"), torch.float64)
    assert ls.interior_op.device == torch.device("cpu")
    assert ls.interior_op.dtype == torch.float64
    assert ls.state.device == torch.device("cpu")
    assert ls.state.dtype == torch.float64
    assert ls.source.device == torch.device("cpu")
    assert ls.source.dtype == torch.float64
