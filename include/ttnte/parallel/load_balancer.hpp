#pragma once

#include "ttnte/mesh/connectivity_graph.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/heuristics.hpp"
#include "ttnte/parallel/parmetis_ops.hpp"
#include "ttnte/parallel/routing_table.hpp"
#include "ttnte/utils/label.hpp"
#include <optional>
#include <string>
#include <torch/types.h>
#include <unordered_map>
#include <vector>

namespace ttnte::parallel {

/// @brief The load balancer runs ParMETIS and METIS for partitioning the mesh
/// (and other data) across MPI ranks for compute optimal balancing.
template<typename BlockType>
class LoadBalancer {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<LoadBalancer>;
  using BlockTypeGID = uint64_t;
  using GIDtoRankMap = std::unordered_map<BlockTypeGID, int>;
  using LoadHeuristicPtr =
    std::shared_ptr<heuristics::LoadHeuristic<BlockType>>;
  using GIDtoNeighborsMap =
    std::unordered_map<BlockTypeGID, std::vector<BlockTypeGID>>;

private:
  // =================================================================
  // Private data
  /// Label.
  Label label_;
  /// The number of MeshBlocks in each MPI rank.
  torch::Tensor block_counts_;

  // =================================================================
  // Private methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::parallel::LoadBalancer<" +
           std::string(BlockType::class_name) + ">::" + func_name;
  }

  /// @brief Compute the weights using the load heuristics passed by the user.
  /// @param load_heuristics A vector of load heuristics.
  /// @return A torch tensor that is shape (n, m) where n is the number of
  /// MeshBlocks on this rank and m is the number of load heuristics.
  torch::Tensor compute_weights(
    const std::vector<LoadHeuristicPtr>& load_heuristics)
  {
    assert(!load_heuristics.empty());

    std::vector<torch::Tensor> weights;
    weights.reserve(load_heuristics.size());

    for (const auto& heuristic : load_heuristics) {
      const torch::Tensor& weight = heuristic->compute_weights();
      TORCH_CHECK(weight.ndimension() == 1,
        "The weights provided by each load heuristic must be 1-D of length\n"
        "equal to the number of local mesh blocks");
      weights.push_back(weight);
    }

    return torch::stack(weights, 1).flatten().contiguous();
  }

  /// @brief Build the routing table for sending MeshBlock and other data based
  /// on the results from the (Par)METIS partitioning.
  /// @param local_gids The global IDs of the MeshBlocks on this MPI rank.
  /// @param partition The MPI rank that each GID should end up at.
  /// @param comm The communicator for MPI communication.
  /// @return The routing table.
  RoutingTable build_routing_table(const torch::Tensor& local_gids,
    const torch::Tensor& partition, const Communicator& comm)
  {
    RoutingTable table;
    int num_blocks = local_gids.size(0);
    int my_rank = comm.rank();
    int world_size = comm.size();

    // Accessors
    auto local_gids_acc = local_gids.template accessor<int64_t, 1>();
    auto partition_acc = partition.template accessor<int64_t, 1>();

    // Populate the send maps
    std::vector<int> send_counts(world_size, 0);

    for (size_t i = 0; i < num_blocks; i++) {
      int target_rank = static_cast<int64_t>(partition_acc[i]);

      if (target_rank != my_rank) {
        table.send_blocks[target_rank].push_back(local_gids_acc[i]);
        send_counts[target_rank]++;
      }
    }

    // Send the number of being send to each other
    std::vector<int32_t> recv_counts(world_size, 0);
    comm.alltoall(send_counts.data(), 1, recv_counts.data(), 1);

    // Actually send the vector of GIDs to each rank we expect to send
    std::vector<Request> requests;

    // Post receives
    for (size_t r = 0; r < world_size; r++) {
      if (recv_counts[r] > 0) {
        table.recv_blocks[r].resize(recv_counts[r]);
        requests.push_back(comm.irecv(table.recv_blocks[r].data(),
          recv_counts[r], r, MPITag::ROUTING_TABLE));
      }
    }

    // Post sends
    for (const auto& [target_rank, gids] : table.send_blocks) {
      requests.push_back(comm.isend(
        gids.data(), gids.size(), target_rank, MPITag::ROUTING_TABLE));
    }

    // Wait for all sends to finish
    Request::wait_all(requests);

    return table;
  }

public:
  // =================================================================
  // Public constructors
  LoadBalancer(int world_size, std::optional<std::string> label = std::nullopt)
    : block_counts_(torch::zeros({world_size}, torch::kInt64)),
      label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}

  // =================================================================
  // Public methods
  /// @brief Compute the initial partition assuming every MPI rank has the
  /// coarse mesh. This uses METIS_PartGraphKway.
  /// @param conn_graph The connectivity graph of the locally held mesh using
  /// the global indices.
  /// @param comm The communicator for MPI communication.
  /// @param load_heuristics A vector of load heuristics for weighting the
  /// partitioning.
  /// @param root_rank The MPI rank to compute the partition which is
  /// broadcasted to all other ranks.
  /// @return A tensor of global IDs to stay on this MPI rank.
  torch::Tensor compute_partition(mesh::ConnectivityGraph conn_graph,
    const parallel::Communicator& comm,
    const std::vector<LoadHeuristicPtr>& load_heuristics = {},
    int root_rank = 0)
  {
    // Check types and device
    TORCH_CHECK(conn_graph.local_gids.is_cpu() && conn_graph.xadj.is_cpu() &&
                  conn_graph.adjncy.is_cpu() && conn_graph.mpi_ranks.is_cpu(),
      "All tensors of the connectivity graph should be on CPU");
    TORCH_CHECK(conn_graph.local_gids.scalar_type() == torch::kInt64 &&
                  conn_graph.xadj.scalar_type() == torch::kInt64 &&
                  conn_graph.adjncy.scalar_type() == torch::kInt64,
      "The `local_gids`, `xadj`, and `adjncy` tensors of the connectivity\n"
      "graph should be 64-bit integer type");
    TORCH_CHECK(conn_graph.mpi_ranks.scalar_type() == torch::kInt32,
      "The `mpi_ranks` tensor should be 32-bit integer type");
    TORCH_CHECK(conn_graph.local_gids.is_contiguous() &&
                  conn_graph.xadj.is_contiguous() &&
                  conn_graph.adjncy.is_contiguous() &&
                  conn_graph.mpi_ranks.is_contiguous(),
      "All tensors in the connectivity graph must be contiguous");

    // Only one MPI rank case
    if (comm.size() == 1) {
      return conn_graph.local_gids;
    }

    int64_t num_blocks = conn_graph.local_gids.size(0);
    int my_rank = comm.rank();
    int world_size = comm.size();
    torch::Tensor partition;

    if (my_rank == root_rank) {
      // Run METIS
      partition = kway_partition(num_blocks, conn_graph.xadj, conn_graph.adjncy,
        load_heuristics.empty() ? torch::ones({num_blocks},
                                    torch::TensorOptions().dtype(torch::kInt64))
                                : compute_weights(load_heuristics),
        load_heuristics.empty() ? 1 : load_heuristics.size(),
        torch::full({static_cast<int64_t>(load_heuristics.size())}, 1.05),
        comm);

    } else {
      // Allocate space for everyone else
      partition = torch::zeros({num_blocks}, conn_graph.local_gids.options());
    }

    // Broadcast the partition to all other ranks
    comm.bcast(partition.template data_ptr<int64_t>(), num_blocks, root_rank);

    // GIDs to keep on this rank
    return conn_graph.local_gids.masked_select(partition == my_rank);
  }

  /// @brief Build the routing table for repartitioning the already distributed
  /// mesh (and other data) using ParMETIS_V3_AdaptiveRepart.
  /// @param conn_graph The connectivity graph of the locally held mesh using
  /// the global indices.
  /// @param comm The communicator for MPI communication.
  /// @param load_heuristics A vector of load heuristics for weighting the
  /// partitioning.
  /// @return The routing table.
  RoutingTable compute_repartition(mesh::ConnectivityGraph conn_graph,
    const parallel::Communicator& comm,
    const std::vector<LoadHeuristicPtr>& load_heuristics = {})
  {
    // Check types and device
    TORCH_CHECK(conn_graph.local_gids.is_cpu() && conn_graph.xadj.is_cpu() &&
                  conn_graph.adjncy.is_cpu() && conn_graph.mpi_ranks.is_cpu(),
      "All tensors of the connectivity graph should be on CPU");
    TORCH_CHECK(conn_graph.local_gids.scalar_type() == torch::kInt64 &&
                  conn_graph.xadj.scalar_type() == torch::kInt64 &&
                  conn_graph.adjncy.scalar_type() == torch::kInt64,
      "The `local_gids`, `xadj`, and `adjncy` tensors of the connectivity\n"
      "graph should be 64-bit integer type");
    TORCH_CHECK(conn_graph.mpi_ranks.scalar_type() == torch::kInt32,
      "The `mpi_ranks` tensor should be 32-bit integer type");
    TORCH_CHECK(conn_graph.local_gids.is_contiguous() &&
                  conn_graph.xadj.is_contiguous() &&
                  conn_graph.adjncy.is_contiguous() &&
                  conn_graph.mpi_ranks.is_contiguous(),
      "All tensors in the connectivity graph must be contiguous");

    // Only one MPI rank case
    if (comm.size() == 1) {
      RoutingTable table;

      table.recv_blocks[comm.rank()] =
        std::vector<int64_t>(conn_graph.local_gids.data_ptr<int64_t>(),
          conn_graph.local_gids.data_ptr<int64_t>() +
            conn_graph.local_gids.numel());
      table.send_blocks[comm.rank()] =
        std::vector<int64_t>(conn_graph.local_gids.data_ptr<int64_t>(),
          conn_graph.local_gids.data_ptr<int64_t>() +
            conn_graph.local_gids.numel());

      return table;
    }

    // Number of local blocks
    int64_t num_blocks = conn_graph.local_gids.size(0);
    auto options = conn_graph.local_gids.options();

    // Gather the number of blocks in each MPI rank
    int my_rank = comm.rank();
    comm.allgather(
      &num_blocks, 1, block_counts_.template data_ptr<int64_t>(), 1);

    // Take partial sum of the block count
    torch::Tensor block_counts =
      torch::zeros({comm.size() + 1}, block_counts_.options());
    block_counts.slice(0, 1) = torch::cumsum(block_counts_, 0);

    // Maps for GID to ParMETIS
    std::unordered_map<int64_t, int64_t> gid2metis;
    std::unordered_map<int, torch::Tensor> rank2ids_send;
    std::unordered_map<int, torch::Tensor> rank2ids_recv;

    // Accessors
    auto local_gids_acc = conn_graph.local_gids.template accessor<int64_t, 1>();
    auto xadj_acc = conn_graph.xadj.template accessor<int64_t, 1>();
    auto adjncy_acc = conn_graph.adjncy.template accessor<int64_t, 1>();
    auto mpi_ranks_acc = conn_graph.mpi_ranks.template accessor<int, 1>();
    std::unordered_map<int,
      std::tuple<size_t, torch::TensorAccessor<int64_t, 1>>>
      rank2ids_acc;

    // Prep map tensor accessor
    {
      // Get the number of unique ranks and then number in each
      const auto& [ranks, inverse_map, counts] =
        torch::_unique2(conn_graph.mpi_ranks, false, false, true);

      // Accessors
      auto ranks_acc = ranks.template accessor<int, 1>();
      auto counts_acc = counts.template accessor<int64_t, 1>();

      // Allocate once
      rank2ids_send.reserve(ranks.size(0));
      rank2ids_recv.reserve(ranks.size(0));
      rank2ids_acc.reserve(ranks.size(0));

      for (size_t i = 0; i < ranks.size(0); i++) {
        int rank = ranks_acc[i];

        // Skip this rank
        if (rank != my_rank) {
          // Where to receive the data
          rank2ids_recv[rank] = torch::zeros({2 * counts_acc[i]}, options);

          // Where to put the data to send
          rank2ids_send[rank] = torch::zeros({2 * counts_acc[i]}, options);
          rank2ids_acc.insert_or_assign(
            rank, std::make_tuple(
                    0, rank2ids_send[rank].template accessor<int64_t, 1>()));
        }
      }
    }

    // Fill maps for GID to ParMETIS
    int64_t base_idx = block_counts[my_rank].item<int64_t>();
    for (size_t i = 0; i < num_blocks; i++) {
      // Our GID
      int64_t gid = local_gids_acc[i];

      // ParMETIS ID
      int64_t pid = i + base_idx;

      // ID maps
      gid2metis[gid] = pid;

      for (size_t j = xadj_acc[i]; j < xadj_acc[i + 1]; j++) {
        if (my_rank != mpi_ranks_acc[j]) {
          auto& [k, ids_acc] = rank2ids_acc.at(mpi_ranks_acc[j]);
          ids_acc[k++] = gid;
          ids_acc[k++] = pid;
        }
      }
    }

    // Non-blocking MPI send of maps
    std::vector<Request> requests;
    requests.reserve(2 * rank2ids_send.size());

    // Post receives
    for (auto& [source_rank, ids_recv] : rank2ids_recv) {
      requests.push_back(comm.irecv(ids_recv.template data_ptr<int64_t>(),
        ids_recv.numel(), source_rank, MPITag::PARTITION_ID_MAP));
    }

    // Post sends
    for (const auto& [target_rank, ids_send] : rank2ids_send) {
      requests.push_back(comm.isend(ids_send.template data_ptr<int64_t>(),
        ids_send.numel(), target_rank, MPITag::PARTITION_ID_MAP));
    }

    // Wait for all to finish
    Request::wait_all(requests);

    // Clear old send map
    rank2ids_send.clear();

    // Populate maps with boundary blocks
    for (const auto& [source_rank, ids_recv] : rank2ids_recv) {
      auto ids_recv_acc = ids_recv.template accessor<int64_t, 1>();

      for (size_t i = 0; i < ids_recv.size(0);) {
        int64_t gid = ids_recv_acc[i++];
        int64_t pid = ids_recv_acc[i++];

        gid2metis[gid] = pid;
      }
    }

    // Clear old recv map
    rank2ids_recv.clear();

    // Map GID to ParMETIS ID
    for (size_t i = 0; i < conn_graph.adjncy.size(0); i++) {
      adjncy_acc[i] = gid2metis.at(adjncy_acc[i]);
    }
    gid2metis.clear();

    // Run ParMETIS adaptive repartition
    torch::Tensor partition = adaptive_repart(block_counts, conn_graph.xadj,
      conn_graph.adjncy,
      load_heuristics.empty()
        ? torch::ones({num_blocks}, torch::TensorOptions().dtype(torch::kInt64))
        : compute_weights(load_heuristics),
      load_heuristics.empty() ? 1 : load_heuristics.size(),
      torch::full({static_cast<int64_t>(load_heuristics.size() * comm.size())},
        1.0 / static_cast<double>(comm.size()),
        torch::TensorOptions().dtype(torch::kFloat64)),
      torch::full({static_cast<int64_t>(load_heuristics.size())}, 1.05,
        torch::TensorOptions().dtype(torch::kFloat64)),
      comm);

    // Create a routing table
    return build_routing_table(conn_graph.local_gids, partition, comm);
  }

  // =================================================================
  // Public getters / setters
  /// @return Get the label of the load balancer.
  const Label& get_label() const noexcept { return label_; }
  /// @return The number of blocks in each MPI rank.
  const torch::Tensor& get_block_counts() const noexcept
  {
    return block_counts_;
  }

  /// @param label The new label of the load balancer.
  void set_label(const std::string& label)
  {
    label_ = Label::from_string(label);
  }
};

} // namespace ttnte::parallel
