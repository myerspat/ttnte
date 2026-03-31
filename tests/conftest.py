import os
import multiprocessing

import pytest
from ttnte import mpi_context


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    multiprocessing.set_start_method("spawn")


@pytest.fixture(scope="session", autouse=True)
def mpi_init():
    """Handle MPI initialization and synchronized tear down for the entire test
    session."""
    mpi_context.init()
