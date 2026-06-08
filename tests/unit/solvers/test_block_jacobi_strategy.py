import torch
import pytest
import torchtt as tntt

from ttnte.solvers import MemoryPolicy, AMEnSolver, BlockJacobiStrategy
from ttnte.linalg import Operator, State, LinearSystem, TTEngine
from ttnte.task import TaskGraph, TaskScheduler
from ttnte.parallel import StreamPool

test_params = [
    (False, MemoryPolicy.OUT_OF_CORE, torch.float32),
    (True, MemoryPolicy.RESIDENT, torch.float32),
    (True, MemoryPolicy.STATE_RESIDENT, torch.float32),
    (True, MemoryPolicy.OPERATOR_RESIDENT, torch.float32),
    (False, MemoryPolicy.OUT_OF_CORE, torch.float64),
    (True, MemoryPolicy.RESIDENT, torch.float64),
    (True, MemoryPolicy.STATE_RESIDENT, torch.float64),
    (True, MemoryPolicy.OPERATOR_RESIDENT, torch.float64),
]


@pytest.mark.parametrize("use_gpu, memory_policy, dtype", test_params[:4])
def test_initialization(use_gpu, memory_policy, dtype):
    # Skip if GPU is requested but not available
    if use_gpu and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    strategy = BlockJacobiStrategy(use_gpu, memory_policy)

    assert strategy.use_gpu == use_gpu
    assert strategy.memory_policy == memory_policy


@pytest.mark.parametrize("use_gpu, memory_policy, dtype", test_params)
def test_build_compute_dag(use_gpu, memory_policy, dtype):
    torch.manual_seed(42)

    # Skip if GPU is requested but not available
    if use_gpu and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create Block-Jacobi DD strategy
    strategy = BlockJacobiStrategy(use_gpu, memory_policy)

    # Create AMEn solver and give it to the straategy
    strategy.set_local_solver(AMEnSolver())

    # Create random matrix and solution
    A = tntt.random([(4, 4), (5, 5), (6, 6), (3, 3)], [1, 2, 3, 2, 1], dtype=dtype)
    b = A @ tntt.random([4, 5, 6, 3], [1, 3, 2, 2, 1], dtype=dtype)
    x0 = tntt.ones([4, 5, 6, 3], dtype=dtype)

    # Run torchTT AMEn solve
    xe = tntt.solvers.amen_solve(A, b, x0=x0, use_cpp=True)

    # Create ttnte linear system
    A = Operator(TTEngine(A.cores))
    x0 = State(TTEngine(x0.cores))
    b = State(TTEngine(b.cores))
    ls = LinearSystem(A, source=b)
    ls.state = x0

    # Create task graph and build the graph for a single local problem
    dag = TaskGraph()
    if use_gpu:
        # Get the device target
        device = torch.device("cuda", 0)

        # Send info to GPU if needed
        if strategy.memory_policy == MemoryPolicy.RESIDENT:
            # Everything remains on the GPU
            xe = xe.to(device)
            ls.to_(device)

        elif strategy.memory_policy == MemoryPolicy.STATE_RESIDENT:
            # Only the state vector remains on the GPU
            xe = xe.to(device)
            ls.transfer_nonbuffer(device)

        elif strategy.memory_policy == MemoryPolicy.OPERATOR_RESIDENT:
            # Only the operators remain on GPU
            ls.transfer_buffer(device)

        h2d_task, _, d2h_task = strategy.build_gpu_compute_dag(
            dag, ls, StreamPool.instance(), True
        )
        assert len(dag) == 3 if strategy.memory_policy != MemoryPolicy.RESIDENT else 1

        if strategy.memory_policy == MemoryPolicy.RESIDENT:
            assert h2d_task == None
            assert d2h_task == None

    else:
        _ = strategy.build_cpu_compute_dag(dag, ls, True)
        assert len(dag) == 1

    # Execute the dag
    scheduler = TaskScheduler()
    scheduler.execute(dag)

    # Get the solution vector
    xa = tntt.TT([core.squeeze(2) for core in ls.state.as_tt().cores])
    assert (
        xa - xe
    ).norm() / xe.norm() < 5e-4  # They won't be the exact same because of random enrichment
