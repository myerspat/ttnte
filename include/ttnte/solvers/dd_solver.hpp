#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/parallel/boundary_communicator.hpp"
#include "ttnte/parallel/communicator.hpp"
#include "ttnte/parallel/stream_pool.hpp"
#include "ttnte/solvers/dd_strategy.hpp"
#include "ttnte/utils/label.hpp"

#include <memory>
#include <optional>
#include <string>

namespace ttnte::solvers {

/// @brief The domain decomposition solver class.
template<typename BlockType>
class DDSolver {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<DDSolver>;
  using Ptr = std::shared_ptr<DDSolver>;
  using MeshPtr = mesh::Mesh<BlockType>::Ptr;
  using SystemPtr = linalg::LinearSystem::Ptr;

private:
  // =================================================================
  // Private data
  /// Label of the DD solver.
  Label label_;
  /// Shared pointer to the mesh.
  MeshPtr mesh_;
  /// All the linear systems on this MPI rank.
  std::vector<SystemPtr> local_systems_;
  /// The solver strategy.
  DDStrategy::Ptr strategy_;

  /// The world communicator.
  parallel::Communicator world_comm_;
  /// The boundary communicators.
  parallel::BoundaryCommunicator boundary_comms_;
  /// The stream pool for GPU streams.
  parallel::StreamPool::Ptr stream_pool_;

  // =================================================================
  // Private constructors
  DDSolver(MeshPtr mesh, DDStrategy::Ptr strategy, int num_streams = 16,
    std::optional<std::string> label = std::nullopt)
    : mesh_(std::move(mesh)), strategy_(std::move(strategy)),
      world_comm_(parallel::Communicator::world()),
      boundary_comms_(world_comm_, 2 * mesh->get_ndim()),
      stream_pool_(parallel::StreamPool::instance(16)),
      label_(label.has_value() ? Label::from_string(label.value())
                               : Label::create_internal())
  {}

public:
  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new instance of the DDSolver.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new DDSolver<BlockType>(std::forward<Args>(args)...));
  }

  // =================================================================
  // Public getters / setters
  /// @return The label of the DD solver.
  const Label& get_label() const noexcept { return label_; }
  /// @return Get the pointer to the DD strategy.
  const DDStrategy::Ptr& get_strategy() const noexcept { return strategy_; }
  /// @return A vector of linear systems for each mesh block local to this MPI
  /// rank.
  const std::vector<SystemPtr>& get_local_systems() const noexcept
  {
    return local_systems_;
  }
  /// @return Get the world communicator.
  const parallel::Communicator& get_world_comm() const noexcept
  {
    return world_comm_;
  }
  /// @return Get the boundary communicators.
  const parallel::BoundaryCommunicator& get_boundary_comms() const noexcept
  {
    return boundary_comms_;
  }
  /// @return Get the GPU stream pool.
  const parallel::StreamPool::Ptr get_stream_pool() const noexcept
  {
    return stream_pool_;
  }

  /// @param local_systems Set the vector of local linear systems for this MPI
  /// rank.
  void set_systems(const std::vector<SystemPtr>& local_systems)
  {
    local_systems_ = local_systems;
  }
};

} // namespace ttnte::solvers
