#pragma once

#include <cstdint>
#include <memory>
#include <torch/extension.h>
#include <vector>

namespace ttnte::sweep {

/// @brief State of one directed edge of the sweep graph for a direction.
enum class EdgeState : uint8_t {
  /// No upwind information crosses the face in this direction.
  NONE,
  /// Upwind dependency: the downstream patch waits for the upstream one.
  DEPENDENCY,
  /// Cycle-breaking lag: the downstream patch reads the upstream face trace
  /// from the previous source iteration.
  LAGGED,
};

/// @brief A face shared by two sweep units (patches), with its per-direction
/// upwind integrals.
struct SweepInterface {
  /// Patch on the owner side of the face.
  int64_t first = 0;
  /// Patch on the other side.
  int64_t second = 0;
  /// (N_Omega) int_face (Omega . n_first)_+ dS: flow first -> second.
  torch::Tensor forward_weight;
  /// (N_Omega) int_face (Omega . n_first)_- dS: flow second -> first.
  torch::Tensor backward_weight;
  /// Weights below this value carry no dependency.
  double tolerance = 0.0;
};

/// @brief Directions sharing one sweep ordering.
struct AngleSet {
  /// (D) int64 direction indices.
  torch::Tensor directions;
  /// Per interface: state of the first -> second edge.
  std::vector<EdgeState> forward;
  /// Per interface: state of the second -> first edge.
  std::vector<EdgeState> backward;
};

/// @brief The work of one sweep task for an angle set: a group of units
/// swept one after another in upwind order.
struct SweepChunk {
  /// Units in topological (upwind-first) order.
  std::vector<int64_t> units;
  /// Indices of the chunks (of the same schedule) this chunk waits for.
  std::vector<int64_t> upstream;
};

/// @brief Per-direction sweep graphs over the local patches, with cycles
/// broken and directions grouped into angle sets.
///
/// Curved IGA faces make the upwind graph cyclic for some directions
/// (Owens et al., JCP 315, 2016). Cycles are broken in two steps:
///  1. For every face with flow in both directions, keep the edge with the
///     larger upwind integral w_ij = int (Omega . n)_+ dS and lag the other.
///     This equals ordering by the face-averaged normal (Owens eq. 48).
///  2. Any cycle that remains is broken by lagging the back edges of a
///     depth-first search.
/// Each patch is solved exactly once per direction per sweep; lagged edges
/// read last iteration's face trace (Owens' "solve once" variant). Directions
/// whose final edge states are identical share a dependency DAG and are
/// swept together as an angle set.
class SweepGraph {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<SweepGraph>;

protected:
  // =================================================================
  // Protected data
  /// Number of sweep units (local patches).
  int64_t num_nodes_ = 0;
  /// Number of directions.
  int64_t num_directions_ = 0;
  /// Interfaces between sweep units.
  std::vector<SweepInterface> interfaces_;
  /// Interfaces touching each node.
  std::vector<std::vector<int64_t>> node_interfaces_;
  /// Angle sets.
  std::vector<AngleSet> angle_sets_;
  /// (N_Omega) angle set of each direction.
  torch::Tensor direction_set_;
  /// Number of edges lagged by the larger-weight rule (summed over
  /// directions).
  int64_t num_weight_lagged_ = 0;
  /// Number of edges lagged by the depth-first search (summed over
  /// directions).
  int64_t num_dfs_lagged_ = 0;

  // =================================================================
  // Protected constructors
  SweepGraph(int64_t num_nodes, std::vector<SweepInterface> interfaces,
    int64_t num_directions);

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new SweepGraph(std::forward<Args>(args)...));
  }

  /// @return Upstream nodes (DEPENDENCY edges into `node`) for an angle set.
  std::vector<int64_t> upstream(size_t angle_set, int64_t node) const;

  /// @return Every node in upwind-first order for an angle set (the graph of
  /// DEPENDENCY edges is acyclic by construction).
  std::vector<int64_t> topological_order(size_t angle_set) const;

  /// @brief Group nodes into sweep tasks for an angle set.
  ///
  /// Nodes sharing a chunk id are swept by one task in topological order.
  /// Grouping can create cycles between chunks even though the node graph is
  /// acyclic (e.g. two patches of FDG elements that exchange flux both ways
  /// along a curved interface); the chunks of each strongly connected
  /// component of the chunk graph are then merged into one task, so any
  /// grouping yields a valid DAG.
  /// @param angle_set The angle set.
  /// @param chunk_of Chunk id of every node (any integers).
  /// @return The tasks, in a topological order of the chunk DAG.
  std::vector<SweepChunk> schedule(
    size_t angle_set, const std::vector<int64_t>& chunk_of) const;

  // =================================================================
  // Public getters
  int64_t num_nodes() const noexcept { return num_nodes_; }
  int64_t num_directions() const noexcept { return num_directions_; }
  const std::vector<SweepInterface>& interfaces() const noexcept
  {
    return interfaces_;
  }
  const std::vector<AngleSet>& angle_sets() const noexcept
  {
    return angle_sets_;
  }
  const torch::Tensor& direction_set() const noexcept { return direction_set_; }
  int64_t num_weight_lagged() const noexcept { return num_weight_lagged_; }
  int64_t num_dfs_lagged() const noexcept { return num_dfs_lagged_; }
};

} // namespace ttnte::sweep
