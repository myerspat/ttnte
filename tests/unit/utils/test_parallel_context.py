import pytest


@pytest.mark.mpi(min_size=2)
def test_parallel_context():
    from ttnte import mpi_context

    # Initialize MPI
    mpi_context.init()

    # Check context world size
    assert mpi_context.world_size >= 2
