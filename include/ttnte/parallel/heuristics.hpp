#pragma once

#include "ttnte/mesh/mesh.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::parallel::heuristics {

template<typename BlockType>
class LoadHeuristic {
protected:
  // =================================================================
  // Protected data
  /// Pointer to the mesh set when finalized.
  const mesh::Mesh<BlockType>* mesh_;

  // State
  bool is_finalized_ = false;

public:
  // =================================================================
  // Public methods
  virtual ~LoadHeuristic() = default;

  /// @brief Computes the weights of the local (specific to this MPI rank)
  /// MeshBlocks for load balancing with ParMETIS.
  virtual torch::Tensor compute_weights() const = 0;

  /// @brief Finalize the heuristic and pass any pointers to needed data
  void finalize(const std::shared_ptr<mesh::Mesh<BlockType>>& mesh_ptr)
  {
    mesh_ = mesh_ptr.get();
    is_finalized_ = true;
  }
};

template<typename BlockType>
class DofHeuristic : public LoadHeuristic<BlockType> {
public:
  /// @brief Computes the weights of the local (specific to this MPI rank)
  /// MeshBlocks for load balancing with ParMETIS. The weights are set to the
  /// number of degrees of freedom for each mesh block.
  torch::Tensor compute_weights() const override
  {
    if (!this->is_finalized_) {
      throw utils::runtime_error(
        "ttnte::parallel::heuristics::DofHeuristic::compute_weights",
        "This heuristic has not been finalized.");
    }
    // Get mesh blocks from the mesh
    const auto& blocks = this->mesh_->get_blocks();

    // Compute the computational weights using the number of DOFs and which
    // patch's have fuel
    torch::Tensor weights =
      torch::zeros({static_cast<int64_t>(blocks.size())}, torch::kInt64);
    auto w_acc = weights.accessor<int64_t, 1>();

    for (size_t i = 0; i < blocks.size(); ++i) {
      w_acc[i] = blocks[i]->get_num_dofs();
    }
    return weights;
  }
};

} // namespace ttnte::parallel::heuristics
