import pytest
import mpi4py.MPI
from ttnte import mpi_context


@pytest.mark.mpi(min_size=2)
def test_parallel_context():
    # Check context world size
    assert mpi_context.world_size >= 2
