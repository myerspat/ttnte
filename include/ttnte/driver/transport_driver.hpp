#pragma once

#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/boundary_communicator.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/heuristics.hpp"
#include "ttnte/parallel/load_balancer.hpp"
#include "ttnte/parallel/parallel_context.hpp"
// #include "ttnte/task/task.hpp"
// #include "ttnte/task/task_graph.hpp"
// #include "ttnte/task/task_scheduler.hpp"
#include "ttnte/utils/label.hpp"
#include "ttnte/xs/server.hpp"
#include <optional>
#include <string>

namespace ttnte::driver {

template<typename BlockType>
class TransportDriver {
  friend parallel::LoadBalancer<BlockType>;

public:
  // =================================================================
  // Public types
  // Core types
  using Label = utils::Label<TransportDriver>;
  using Ptr = std::shared_ptr<TransportDriver>;
  using MeshPtr = std::shared_ptr<mesh::Mesh<BlockType>>;
  using XSServerPtr = std::shared_ptr<xs::Server>;

  // Communication and load balancing
  using Communicator = parallel::Communicator;
  using BoundaryCommunicator = parallel::BoundaryCommunicator;
  using LoadHeuristicPtr =
    std::shared_ptr<parallel::heuristics::LoadHeuristic<BlockType>>;
  using LoadBalancer = parallel::LoadBalancer<BlockType>;

  // // Tasked-based types
  // using TaskStatus = task::TaskStatus;
  // using DeviceTarget = task::DeviceTarget;
  // using TaskGraph = task::TaskGraph;
  // using TaskSchedulter = task::TaskScheduler;

private:
  // =================================================================
  // Private data
  /// Label
  Label label_;
  /// Shared pointer to the mesh
  MeshPtr mesh_;
  /// Shared pointer to the cross section server
  XSServerPtr xs_server_;

  // Communication and load balancing
  /// Communicator responsible for MPI communication
  Communicator comm_;
  /// Load balancer for running (Par)METIS
  LoadBalancer load_balancer_;

  // States
  bool is_distributed_ = false;

  // =================================================================
  // Private constructors
  TransportDriver(MeshPtr mesh, XSServerPtr xs_server,
    const parallel::ParallelContext& mpi_context,
    std::optional<std::string> label = std::nullopt)
    : mesh_(std::move(mesh)), xs_server_(std::move(xs_server)),
      comm_(Communicator::world()), load_balancer_(mpi_context.world_size()),
      label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {
    // Finalize the mesh and server if not already finalized
    if (!mesh_->is_finalized()) {
      mesh_->finalize();
    }
    if (!xs_server_->is_finalized()) {
      xs_server_->finalize();
    }
  }

public:
  // =================================================================
  // Public methods
  /// @brief Build a TransportDriver and get the shared pointer to it.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new TransportDriver<BlockType>(std::forward<Args>(args)...));
  }

  /// @brief Partition the Mesh according to the load heuristics. Partitioning
  /// runs METIS_PartGraphKway on rank root_rank.
  /// @param load_heuristics These heuristics compute weights for each local
  /// MeshBlock to help METIS decide the best repartition.
  /// @param root_rank The rank to run METIS on.
  void distribute(
    std::vector<LoadHeuristicPtr> load_heuristics = {}, int root_rank = 0)
  {
    // MPI world size is one (only one MPI rank)
    if (comm_.size() == 1) {
      return;
    }

    // Finalize the heuristics
    for (auto& heuristic : load_heuristics) {
      heuristic->finalize(mesh_);
    }

    // Determine what MeshBlocks will stay on this MPI rank
    const auto& local_gids = load_balancer_.compute_partition(
      std::move(mesh_->build_connectivity_graph()), comm_, load_heuristics,
      root_rank);

    // Restrict the mesh
    mesh_->cull_blocks(local_gids);
    is_distributed_ = true;
  }

  /// @brief Repartition the Mesh according to the load heuristics.
  /// Repartitioning runs ParMETIS_V3_AdaptiveRepart
  /// (https://karypis.github.io/glaros/files/sw/parmetis/manual.pdf).
  /// @param load_heuristics These heuristics compute weights for each local
  /// MeshBlock to help ParMETIS decide the best repartition.
  void redistribute(std::vector<LoadHeuristicPtr> load_heuristics = {})
  {
    // MPI world size is one (only one MPI rank)
    if (comm_.size() == 1) {
      return;
    }

    // Finalize the heuristics
    for (auto& heuristic : load_heuristics) {
      heuristic->finalize(mesh_);
    }

    // Create routing table for sending MeshBlocks around
    const auto& routing_table = load_balancer_.compute_repartition(
      std::move(mesh_->build_connectivity_graph()), comm_, load_heuristics);
  }

  // =================================================================
  // Public getters / setters
  /// @return Get the label of the driver.
  const Label& get_label() const noexcept { return label_; }
  /// @return Get the shared pointer to the mesh.
  const MeshPtr& get_mesh() const noexcept { return mesh_; }
  /// @return Get the shared pointer to the cross section library.
  const XSServerPtr& get_server() const noexcept { return xs_server_; }

  /// @param label The new label of the driver.
  void set_label(const std::string& label)
  {
    label_ = Label::from_string(label);
  }
};

} // namespace ttnte::driver
