import pytest
import torch
import torchtt as tntt

from ttnte.linalg import State, Operator, LinearSystem, TTEngine
from ttnte.solvers import AMEnSolver

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]


@pytest.mark.parametrize("device, dtype", test_params)
def test_amen_solver(device, dtype):
    torch.manual_seed(42)

    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create random matrix and solution
    A = tntt.random(
        [(4, 4), (5, 5), (6, 6), (3, 3)], [1, 2, 3, 2, 1], dtype=dtype, device=device
    )
    b = A @ tntt.random([4, 5, 6, 3], [1, 3, 2, 2, 1], dtype=dtype, device=device)
    x0 = tntt.ones([4, 5, 6, 3], dtype=dtype, device=device)

    # Run torchTT AMEn solve
    xe = tntt.solvers.amen_solve(A, b, x0=x0, use_cpp=True)

    A = Operator(TTEngine(A.cores))
    x0 = State(TTEngine(x0.cores))
    b = State(TTEngine(b.cores))
    ls = LinearSystem(A, source=b)
    ls.state = x0

    # Run AMEnSolver
    solver = AMEnSolver()
    solver.solve(ls)

    # Get the solution vector
    xa = tntt.TT([core.squeeze(2) for core in ls.state.as_tt().cores])
    assert (
        xa - xe
    ).norm() / xe.norm() < 5e-4  # They won't be the exact same because of random enrichment
