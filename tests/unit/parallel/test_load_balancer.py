import pytest
import torch

from ttnte import mpi_context
from ttnte.parallel import IGALoadBalancer, IGALoadHeuristic, Communicator
from ttnte.mesh import ConnectivityGraph

# Initialize MPI
mpi_context.init()


class LoadHeuristicTesting(IGALoadHeuristic):
    """This is a custom class for testing."""

    def __init__(self, weights):
        super().__init__()
        self._weights = weights

    def compute_weights(self):
        return self._weights


def test_initialize():
    # Initialize the load balancer
    load_balancer = IGALoadBalancer(mpi_context.world_size, "Load Balancer")

    assert load_balancer.label.to_string() == "Load Balancer"
    torch.testing.assert_close(
        load_balancer.block_counts,
        torch.zeros((mpi_context.world_size), dtype=torch.int64),
    )


@pytest.mark.mpi(min_size=2)
def test_compute_partition():
    device = "cpu"
    dtype = torch.int64

    # Check MPI size
    if mpi_context.world_size > 2:
        pytest.skip("Test requires exactly 2 processes")

    # Create a made up connectivity graph for this mesh
    #  _______ _______
    # |       |       |
    # |   0   |   1   |
    # |_______|_______|
    # |       |       |
    # |   2   |   3   |
    # |_______|_______|
    conn_graph = ConnectivityGraph()
    conn_graph.local_gids = torch.tensor([0, 1, 2, 3], dtype=dtype, device=device)
    conn_graph.xadj = torch.tensor([0, 2, 4, 6, 8], dtype=dtype, device=device)
    conn_graph.adjncy = torch.tensor(
        [1, 2, 0, 3, 0, 3, 1, 2], dtype=dtype, device=device
    )
    conn_graph.mpi_ranks = mpi_context.rank * torch.ones(
        4, dtype=torch.int32, device=device
    )

    # Create load heuristic
    # Make the weights [2, 2, 1, 1] therefore we expect the split to be [0, 2] and [1, 3]
    heuristic = LoadHeuristicTesting(
        torch.tensor([2, 2, 1, 1], dtype=dtype, device=device)
    )

    # Create the communicator
    comm = Communicator.world()

    # Build load balancer
    load_balancer = IGALoadBalancer(comm.size())

    # Compute the initial partition
    local_gids = load_balancer.compute_partition(conn_graph, comm, [heuristic])

    # 1. Define the two valid expected partitions
    expected_a = torch.tensor([0, 2], dtype=dtype, device=device)
    expected_b = torch.tensor([1, 3], dtype=dtype, device=device)

    # 2. Sort the returned GIDs just in case METIS returns them out of order (e.g., [2, 0])
    local_gids_sorted, _ = torch.sort(local_gids)

    # 3. Check for exact equality with either valid partition
    matches_a = torch.equal(local_gids_sorted, expected_a)
    matches_b = torch.equal(local_gids_sorted, expected_b)

    # 4. Assert that it matches one of them
    assert (
        matches_a or matches_b
    ), f"Rank {comm.rank()} got an invalid partition: {local_gids}"


@pytest.mark.mpi(min_size=2)
def test_compute_repartition():
    device = "cpu"
    dtype = torch.int64

    # Check MPI size
    if mpi_context.world_size > 2:
        pytest.skip("Test requires exactly 2 processes")

    # Create a made up connectivity graph for this mesh
    #  _______ _______
    # |       |       |
    # |   0   |   1   |
    # |_______|_______|
    # |       |       |
    # |   2   |   3   |
    # |_______|_______|
    conn_graph = ConnectivityGraph()
    heuristic = None
    if mpi_context.rank == 0:
        conn_graph.local_gids = torch.tensor([2, 3], dtype=dtype, device=device)
        conn_graph.xadj = torch.tensor([0, 2, 4], dtype=dtype, device=device)
        conn_graph.adjncy = torch.tensor([0, 3, 1, 2], dtype=dtype, device=device)
        conn_graph.mpi_ranks = torch.tensor(
            [1, 0, 1, 0], dtype=torch.int32, device=device
        )

        # Create load heuristic
        # Make the weights [2, 2, 1, 1] therefore we expect the split to be [0, 2] and [1, 3]
        heuristic = LoadHeuristicTesting(
            torch.tensor([1, 1], dtype=dtype, device=device)
        )
    else:
        conn_graph.local_gids = torch.tensor([0, 1], dtype=dtype, device=device)
        conn_graph.xadj = torch.tensor([0, 2, 4], dtype=dtype, device=device)
        conn_graph.adjncy = torch.tensor([1, 2, 0, 3], dtype=dtype, device=device)
        conn_graph.mpi_ranks = torch.tensor(
            [1, 0, 1, 0], dtype=torch.int32, device=device
        )

        # Create load heuristic
        # Make the weights [2, 2, 1, 1] therefore we expect the split to be [0, 2] and [1, 3]
        heuristic = LoadHeuristicTesting(
            torch.tensor([2, 2], dtype=dtype, device=device)
        )

    # Create the communicator
    comm = Communicator.world()

    # Build load balancer
    load_balancer = IGALoadBalancer(comm.size())

    # Compute the initial partition
    routing_table = load_balancer.compute_repartition(conn_graph, comm, [heuristic])

    # Convert our starting local_gids to a Python set for easy math
    final_gids = set(conn_graph.local_gids.cpu().tolist())

    # Subtract the blocks we are sending away
    for _, blocks in routing_table.send_blocks.items():
        for block_id in blocks:
            # Note: Depending on pybind11 casting, block_id might be an int or a tensor scalar
            final_gids.remove(int(block_id))

    # Add the blocks we are receiving
    for _, blocks in routing_table.recv_blocks.items():
        for block_id in blocks:
            final_gids.add(int(block_id))

    # Define the valid expected states
    expected_partitions = [{0, 2}, {1, 3}]

    # 5. Assert that our final local GIDs match one of the expected partitions
    assert (
        final_gids in expected_partitions
    ), f"Rank {mpi_context.rank} failed! Ended up with {final_gids}"
