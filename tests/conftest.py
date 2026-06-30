import multiprocessing

import torch
import pytest

from ttnte import mpi_context


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    multiprocessing.set_start_method("spawn")


@pytest.fixture(scope="session", autouse=True)
def mpi_init():
    """Handle MPI initialization and synchronized tear down for the entire test
    session."""
    # Initialize MPI
    mpi_context.init()
    num_threads_per_rank = min(
        multiprocessing.cpu_count() // mpi_context.world_size, 16
    )
    torch.set_num_threads(num_threads_per_rank)

    # Initialize CUDA if needed
    if torch.cuda.is_available():
        _ = torch.linalg.qr(torch.zeros(2, 2, device="cuda"))
