#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/mesh/mesh_block.hpp"
#include "ttnte/physics/assembly_configs.hpp"
#include <memory>

namespace ttnte::physics {

template<typename BlockType, typename ConfigType>
class DGAssembler {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<DGAssembler>;

protected:
  // =================================================================
  // Protected data
  /// Configuration for assembly.
  ConfigType config_;
  /// Pointer to the mesh block.
  mesh::MeshBlock<BlockType>::Ptr block_;
  /// Pointer to the linear system after assembly.
  linalg::LinearSystem::Ptr linear_system_;

  // =================================================================
  // Protected constructors
  DGAssembler(const mesh::MeshBlock<BlockType>::Ptr& block,
    const DGAssemblerConfig& config)
    : block_(block), config_(config)
  {}

public:
  virtual ~DGAssembler() = default;

  // =================================================================
  // Public methods
  /// @brief Assemble the linear system for this mesh block.
  /// @return The full linear system object.
  virtual linalg::LinearSystem::Ptr assemble() = 0;

  // =================================================================
  // Public getters / setters
  /// @return The assembly configuration.
  const DGAssemblerConfig& get_config() const noexcept { return config_; }
  /// @return The mesh block for this assembler.
  const mesh::MeshBlock<BlockType>::Ptr get_block() const noexcept
  {
    return block_;
  }
  /// @return The assembled linear system.
  const linalg::LinearSystem::Ptr& get_linear_system() const noexcept
  {
    return linear_system_;
  }
};

} // namespace ttnte::physics
