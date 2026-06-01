#pragma once

#include "ttnte/mesh/connectivity_graph.hpp"
#include "ttnte/mesh/mesh_block.hpp"
#include "ttnte/mesh/mesh_block_boundary.hpp"
#include <ATen/Parallel.h>
#include <algorithm>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <torch/types.h>
#include <tuple>

namespace ttnte::mesh {

template<typename BlockType>
class Mesh {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<Mesh>;
  using Ptr = std::shared_ptr<Mesh>;
  using BlockTypeGID = uint64_t;
  using BlockTypePtr = MeshBlock<BlockType>::Ptr;
  using MeshBlocks = std::vector<BlockTypePtr>;
  using GIDtoIdx = std::unordered_map<BlockTypeGID, size_t>;

  static constexpr const char* class_name = "Mesh";

private:
  // =================================================================
  // Private data
  /// The label of the mesh.
  Label label_;
  /// The vector of MeshBlocks.
  MeshBlocks blocks_;
  /// The number of blocks globally (between all ranks).
  size_t global_num_blocks_ = 0;
  /// The global bounding box of the mesh.
  torch::Tensor bbox_;
  /// Global ID to vector index map.
  GIDtoIdx index_map_;
  /// MPI rank of this mesh.
  int my_rank_;

  // States
  bool is_connected_ = false;
  bool is_finalized_ = false;

  // Thread lock
  std::mutex mesh_mutex;

  // =================================================================
  // Private methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return "ttnte::mesh::" + std::string(class_name) + "::" + func_name;
  }
  void is_finalized_or_error(const std::string& func_name) const
  {
    if (!is_finalized_) {
      throw utils::runtime_error(*this, func_name,
        "This class must be finalized before running this method");
    }
  }
  void is_connected_or_error(const std::string& func_name) const
  {
    if (!is_connected_) {
      throw utils::runtime_error(*this, func_name,
        "This class must be connected before running this method");
    }
  }
  void is_not_finalized_or_error(const std::string& func_name) const
  {
    if (is_finalized_) {
      throw utils::runtime_error(
        *this, func_name, "This class has already been finalized");
    }
  }
  void is_not_connected_or_error(const std::string& func_name) const
  {
    if (is_connected_) {
      throw utils::runtime_error(
        *this, func_name, "This class has already been connected");
    }
  }

  void check_added_blocks(const std::string& method_name) const
  {
    if (blocks_.empty()) {
      throw utils::runtime_error(*this, error_context(method_name),
        "No MeshBlocks were added to the mesh");
    }

    // Check each block has been finalized and if not we finalize them
    for (const auto& bptr : blocks_) {
      if (!bptr->is_finalized()) {
        bptr->finalize();
      }
    }

    // Make sure all blocks are on the same device with the same data type and
    // same dimensionality
    const auto& dtype = blocks_[0]->get_dtype();
    const auto& device = blocks_[0]->get_device();
    const auto& ndim = blocks_[0]->get_ndim();

    for (const auto& bptr : blocks_) {
      if (bptr->get_dtype() != dtype || bptr->get_device() != device) {
        throw utils::runtime_error(*this, error_context(method_name),
          "All blocks must be on the same device with the same data type");
      }
      if (bptr->get_ndim() != ndim) {
        throw utils::runtime_error(*this, error_context(method_name),
          "All blocks must have the same number of dimensions");
      }
    }
  }

  // =================================================================
  // Private constructors
  Mesh(const parallel::ParallelContext& mpi_context,
    std::optional<std::string> label = std::nullopt)
    : my_rank_(mpi_context.rank()),
      label_(label.has_value() ? Label::from_string(*label)
                               : Label::create_internal())
  {}
  Mesh(int my_rank, std::optional<std::string> label = std::nullopt)
    : my_rank_(my_rank), label_(label.has_value() ? Label::from_string(*label)
                                                  : Label::create_internal())
  {}

public:
  // =================================================================
  // Public methods
  /// @brief Create a new Mesh and return a shared pointer to it.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return std::shared_ptr<Mesh<BlockType>>(
      new Mesh<BlockType>(std::forward<Args>(args)...));
  }

  /// @return Is the Mesh immutable?
  bool is_finalized() const noexcept { return is_finalized_; }
  /// @return Has the MeshBlocks in the mesh been connected?
  bool is_connected() const noexcept { return is_connected_; }
  /// @brief Run a final check on the mesh, set all unknown boundary conditions
  /// to vacuum, and set the Mesh to immutable. This can be called when the mesh
  /// has been connected but has yet to be finalized
  void finalize()
  {
    // Lock class from multiple threads calling
    std::lock_guard<std::mutex> lock(mesh_mutex);

    is_connected_or_error("finalize");
    is_not_finalized_or_error("finalize");

    // Iterate through the remaining unknown boundary types and set them as
    // vacuum
    for (size_t i = 0; i < blocks_.size(); i++) {
      const auto& bptr = blocks_[i];

      // Add a global ID
      bptr->set_gid(i);
      index_map_[i] = i;

      for (size_t dim = 0; dim < blocks_[0]->get_ndim(); dim++) {
        for (bool is_upper : {false, true}) {
          if (bptr->get_boundary_info(dim, is_upper).get_type() ==
              physics::BoundaryType::UNKNOWN) {
            bptr->set_boundary_type(
              dim, is_upper, physics::BoundaryType::VACUUM);
          }
        }
      }
    }

    is_finalized_ = true;
  };

  /// @brief Reserve positions in the MeshBlock vector. This can be called
  /// before the mesh has been connected and finalized.
  /// @param size The new maximum number of MeshBlocks the vector can hold.
  void reserve(const size_t& size)
  {
    // Lock class from multiple threads calling
    std::lock_guard<std::mutex> lock(mesh_mutex);

    is_not_connected_or_error("reserve");
    is_not_finalized_or_error("reserve");

    blocks_.reserve(size);
  }

  /// @brief Add a new MeshBlock to the Mesh. This can be called
  /// before the mesh has been connected and finalized.
  /// @param new_block The shared pointer to the new mesh block.
  void add_block(BlockTypePtr new_block)
  {
    // Lock class from multiple threads calling
    std::lock_guard<std::mutex> lock(mesh_mutex);

    is_not_connected_or_error("add_block");
    is_not_finalized_or_error("add_block");

    blocks_.push_back(new_block);

    // Set the number of blocks
    global_num_blocks_ = blocks_.size();
  }

  /// @brief Connect the mesh blocks to each other. We build an undirected graph
  /// of boundaries each pointing to their neighboring MeshBlock. Note this
  /// assumes no hanging nodes at the coarsest level (assumed to be what is
  /// already in the vector of MeshBlocks). Here we calculate the global
  /// bounding box and populate the boundary vectors of each MeshBlock. We use
  /// the center of the bounding box for each MeshBlock to determine the initial
  /// connectivity and then check the MeshBlock coordinates at each face. We
  /// compute any flips or permutations for ghost boundary passing. This may be
  /// called once before the mesh is finalized. Ensure all MeshBlocks have been
  /// added first.
  /// @param tol The tolerance for the geometric checks.
  void connect(const double& tol = 1e-8)
  {
    // Lock class from multiple threads calling
    std::lock_guard<std::mutex> lock(mesh_mutex);

    // State checks
    is_not_connected_or_error("connect");
    is_not_finalized_or_error("connect");
    check_added_blocks("connect");

    // Calculate the global bounding box
    {
      auto it = blocks_.cbegin();
      bbox_ = (*it)->get_bbox().to(torch::kFloat64);
      auto bbox_min = bbox_[0];
      auto bbox_max = bbox_[1];
      it++;

      while (it != blocks_.cend()) {
        const auto& bbox = (*it)->get_bbox();

        bbox_min = torch::min(bbox_min, bbox[0]);
        bbox_max = torch::max(bbox_max, bbox[1]);
        it++;
      }
      bbox_ = torch::stack({bbox_min, bbox_max}, 0);
    }

    // Hash for the boundaries with matching centroids
    std::unordered_map<PointKey,
      std::vector<std::tuple<size_t, uint64_t, bool>>, PointHash>
      spatial_hash;
    spatial_hash.reserve(blocks_.size() * blocks_[0]->get_ndim() * 2);

    // Iterate through each mesh block face, find the centroid, add it to the
    // hash
    for (size_t i = 0; i < blocks_.size(); i++) {
      const auto& block = blocks_[i];
      for (size_t dim = 0; dim < block->get_ndim(); dim++) {
        for (bool is_upper : {false, true}) {
          // Get the boundary mesh block
          const auto& bblock = block->get_boundary(dim, is_upper);

          // Get the bounding box center
          const torch::Tensor& bbox = bblock->get_bbox().to(torch::kFloat64);
          const torch::Tensor bbox_center = bbox.sum(0) / 2.0;

          // Check if the face is degenerate and set it to be so
          if (is_face_degenerate(bblock->get_coords(), tol)) {
            block->set_boundary_type(
              dim, is_upper, physics::BoundaryType::DEGENERATE);
          } else {
            // Get the key for this
            auto key = PointKey(bbox_center, tol);
            spatial_hash[key].push_back(std::make_tuple(i, dim, is_upper));
          }
        }
      }
    }

    // Find the neighbors for each boundary (if internal)
    for (auto it = spatial_hash.begin(); it != spatial_hash.end(); it++) {
      // The faces for this point
      const auto& faces = it->second;

      // Check if this face is on a boundary with no matches
      if (faces.size() == 1) {
        continue;
      }

      // Given the coarse mesh must have no hanging nodes we expect each
      // face has exactly two mesh blocks
      bool found_match = false;
      size_t i = 0;

      do {
        if (i >= faces.size()) {
          break;
        }

        const auto& [block_a_idx, dim_a, is_upper_a] = faces[i];

        // Get the main MeshBlock
        auto& block_a = blocks_[block_a_idx];
        const auto& bblock_a = block_a->get_boundary(dim_a, is_upper_a);

        // Iterate through the other faces
        for (size_t j = i + 1; j < faces.size(); j++) {
          const auto& [block_b_idx, dim_b, is_upper_b] = faces[j];

          // Get this block and its boundary
          auto& block_b = blocks_[block_b_idx];
          const auto& bblock_b = block_b->get_boundary(dim_b, is_upper_b);

          // Call boundary mapping which is a vector of true/false info
          // For D dimensions:
          const auto& [is_coupled, mapping_a, mapping_b] =
            get_boundary_mapping(bblock_a, bblock_b, tol);

          if (is_coupled) {
            const size_t face_a_id =
              dim_a * 2 + static_cast<size_t>(is_upper_a);
            const size_t face_b_id =
              dim_b * 2 + static_cast<size_t>(is_upper_b);

            // Set the mapping and neighbor information for each boundary
            block_a->add_connection(dim_a, is_upper_a,
              {static_cast<int64_t>(block_b_idx), face_b_id, my_rank_,
                mapping_a});
            block_b->add_connection(dim_b, is_upper_b,
              {static_cast<int64_t>(block_a_idx), face_a_id, my_rank_,
                mapping_b});

            // We found a match
            found_match = true;
            break;
          }
        }
        i++;
      } while (!found_match);
    }

    is_connected_ = true;
  }

  /// @brief Check if a boundary is degenerate. An example of a degenerate face
  /// is the corner of a quarter circle NURBS patch. The boundary is isolated to
  /// a single point and thus its area is zero. For degenerate boundaries we
  /// treat them as vacuum boundaries.
  /// @param coords The mesh coordinates.
  /// @param tol The tolerance for geometric checks.
  /// @return Whether the face is degenerate or not.
  static bool is_face_degenerate(const torch::Tensor& coords, double tol = 1e-8)
  {
    // Determine topological dimension (grid_ndim)
    // coords shape is [N1, ..., ND, spatial_dims]
    int64_t tensor_ndim = coords.ndimension();
    int64_t grid_ndim = tensor_ndim - 1;

    // Flatten the grid so we just have a list of points: [Total_Points,
    // Spatial_Dims]
    torch::Tensor pts = coords.reshape({-1, coords.size(-1)});

    // Center the points around their mean
    torch::Tensor center = pts.mean(0);
    torch::Tensor centered_pts = pts - center;

    // Perform SVD to find the physical rank
    // linalg_svd returns a tuple: (U, S, Vh). We only need the singular values
    // (S).
    const auto& s = torch::linalg_svdvals(centered_pts);

    // Count how many singular values are significantly larger than zero
    // This tells us how many physical dimensions the points actually span.
    int physical_rank = (s > tol).sum().item<int>();

    // If the physical rank is less than the grid topology, it has collapsed.
    return physical_rank < grid_ndim;
  }

  /// @brief Check if the two faces match and determine if the dimensions must
  /// permute or flip to make this work.
  /// @param face_a The first boundary represented as another MeshBlock.
  /// @param face_b The second boundary represented as another MeshBlock.
  /// @return A tuple where the first element is a Boolean of whether the two
  /// boundaries match, the second is the mapping for data received from face_b
  /// going to face_a, and the third is the opposite of the first mapping.
  static std::tuple<bool, BoundaryMapping, BoundaryMapping>
  get_boundary_mapping(const BlockTypePtr& face_a, const BlockTypePtr& face_b,
    const double& tol = 1e-8)
  {
    // Boundary mappings to fill
    BoundaryMapping mapping_a;
    BoundaryMapping mapping_b;

    // Check they have the same number of elements, dimensions, and their
    // bounding box matches
    if (face_a->get_numel() != face_b->get_numel() ||
        face_a->get_ndim() != face_b->get_ndim() ||
        !torch::allclose(face_a->get_bbox(), face_b->get_bbox(), tol, tol)) {
      return std::make_tuple(false, mapping_a, mapping_b);
    }
    int64_t ndim = face_a->get_ndim();

    // Get mesh points (N1, ..., ND, xyz)
    torch::Tensor coords_a = face_a->get_coords();
    torch::Tensor coords_b = face_b->get_coords();

    // Permutation generator
    c10::SmallVector<int64_t, 2> p(ndim);
    std::iota(p.begin(), p.end(), 0);

    // Search all permutations possible
    do {
      // Search all possible flip states
      int num_flip_states = 1 << ndim;
      for (int flip_state = 0; flip_state < num_flip_states; flip_state++) {
        c10::SmallVector<int64_t, 2> flip_dims;
        c10::SmallVector<bool, 2> current_flips(ndim, false);

        // Decode the bitmask into actual tensor dimensions to flip
        for (int d = 0; d < ndim; ++d) {
          if ((flip_state >> d) & 1) {
            flip_dims.push_back(d);
            current_flips[d] = true;
          }
        }

        // Apply mapping to the other face
        c10::SmallVector<int64_t, 3> full_p(p.begin(), p.end());
        full_p.push_back(ndim);
        torch::Tensor test_b = coords_b.permute(full_p);
        if (!flip_dims.empty()) {
          test_b = test_b.flip(flip_dims);
        }

        // Check for an exact match
        if (torch::allclose(coords_a, test_b, tol, tol)) {
          // How face A reads face B's ghost data
          mapping_a.perm = full_p;
          mapping_a.flip = current_flips;

          // How face B reads face A's ghost data (inverse)
          c10::SmallVector<int64_t, 3> p_inv(ndim);
          c10::SmallVector<bool, 2> f_inv(ndim, false);

          for (int i = 0; i < ndim; ++i) {
            p_inv[p[i]] = i;                // Inverse permutation
            f_inv[p[i]] = current_flips[i]; // Inverse flip
          }
          p_inv.push_back(full_p.back());

          mapping_b.perm = p_inv;
          mapping_b.flip = f_inv;

          return std::make_tuple(true, mapping_a, mapping_b);
        }
      }
    } while (std::next_permutation(p.begin(), p.end()));

    // No matches found
    return std::make_tuple(false, mapping_a, mapping_b);
  }

  /// @brief Set the boundary conditions along the minimum and/or maximum of
  /// some prescribed dimensions. This assumes the boundaries are along
  /// Euclidean axes.
  /// @param bcplanes Which planes are set with the boundary condition.
  /// @param type The boundary condition type.
  /// @param tol The tolerance for geometric comparisons.
  void set_axis_aligned_conditions(const physics::BCPlane& bcplanes,
    const physics::BoundaryType& type, double tol = 1e-8)
  {
    // Lock class from multiple threads calling
    std::lock_guard<std::mutex> lock(mesh_mutex);

    // State checks
    is_connected_or_error("set_axis_aligned_condition");
    is_not_finalized_or_error("set_axis_aligned_condition");

    // Global bbox accessor
    auto bbox_acc = bbox_.accessor<double, 2>();
    const auto& active_planes = bcplanes.get_active_planes();

    // Iterate through the blocks and find those that haven't been assigned yet
    for (const auto& bptr : blocks_) {
      for (size_t dim = 0; dim < bptr->get_ndim(); dim++) {
        for (bool is_upper : {false, true}) {
          // Check the current boundary type
          if (bptr->get_boundary_info(dim, is_upper).get_type() ==
              physics::BoundaryType::UNKNOWN) {
            // Check if this boundary is on the BCPlane
            const auto& bbox =
              bptr->get_boundary(dim, is_upper)->get_bbox().to(torch::kFloat64);
            const torch::Tensor bbox_center = bbox.sum(0) / 2.0;

            for (size_t d = 0; d < bbox_center.size(0); d++) {
              const size_t i = d * 2;

              if (i < active_planes.size() && active_planes[i] &&
                  std::abs(bbox_center[d].item<double>() - bbox_acc[0][d]) <
                    tol) {
                bptr->set_boundary_type(dim, is_upper, type);
              } else if (i + 1 < active_planes.size() && active_planes[i + 1] &&
                         std::abs(bbox_center[d].item<double>() -
                                  bbox_acc[1][d]) < tol) {
                bptr->set_boundary_type(dim, is_upper, type);
              }
            }
          }
        }
      }
    }
  }

  /// @return The local connectivity graph (which MeshBlocks talk
  /// to who) using global indices.
  ConnectivityGraph build_connectivity_graph() const
  {
    is_connected_or_error("build_connectivity_graph");
    const auto& options = torch::TensorOptions().dtype(torch::kInt64);
    int64_t num_blocks = blocks_.size();

    // Create connectivity graph
    ConnectivityGraph graph;
    graph.local_gids = torch::empty({num_blocks}, options);
    graph.xadj = torch::zeros({num_blocks + 1}, options);

    // Accessors
    auto local_gids_acc = graph.local_gids.accessor<int64_t, 1>();
    auto xadj_acc = graph.xadj.accessor<int64_t, 1>();

    // First pass over all connections
    for (size_t i = 0; i < num_blocks; i++) {
      const auto& bptr = blocks_[i];
      xadj_acc[i + 1] = xadj_acc[i];

      // Save the global ID and continue partial sum
      local_gids_acc[i] = bptr->get_gid();

      for (const auto& boundary : bptr->get_boundary_info()) {
        if (boundary.get_type() == physics::BoundaryType::INTERNAL) {
          // Get and add connections
          xadj_acc[i + 1] += boundary.get_connections().size();
        }
      }
    }

    // Allocate adjacency vector
    graph.adjncy =
      torch::empty({graph.xadj[num_blocks].item<int64_t>()}, options);
    graph.mpi_ranks = torch::empty({graph.xadj[num_blocks].item<int64_t>()},
      torch::TensorOptions().dtype(torch::kInt32));

    // Accessors
    auto adjncy_acc = graph.adjncy.accessor<int64_t, 1>();
    auto mpi_ranks_acc = graph.mpi_ranks.accessor<int32_t, 1>();

    // Iterate through all the boundaries again to collect the adjacency list
    // and rank to ID mappings
    for (size_t i = 0; i < num_blocks; i++) {
      const auto& bptr = blocks_[i];

      // xadj_acc[i] gives the starting point in adjncy
      int64_t offset = xadj_acc[i];

      for (const auto& boundary : bptr->get_boundary_info()) {
        if (boundary.get_type() == physics::BoundaryType::INTERNAL) {
          for (const auto& connection : boundary.get_connections()) {
            // Add adjacent GID and MPI rank
            adjncy_acc[offset] = connection.gid;
            mpi_ranks_acc[offset] = connection.mpi_rank;
            offset++;
          }
        }
      }
    }

    return graph;
  }

  /// @brief Remove any blocks who's global ID (GID) does not exist in
  /// keep_gids.
  /// @param keep_gids GIDs of the blocks we plan to keep.
  void cull_blocks(const torch::Tensor& keep_gids)
  {
    TORCH_CHECK(
      keep_gids.ndimension() == 1 && keep_gids.size(0) <= blocks_.size(),
      "`keep_gids` must be 1-D and not longer than then number of blocks in\n"
      "the mesh");
    MeshBlocks new_blocks_;
    GIDtoIdx new_index_map_;
    new_blocks_.reserve(keep_gids.size(0));
    new_index_map_.reserve(keep_gids.size(0));

    // Accessor
    auto keep_gids_acc = keep_gids.accessor<int64_t, 1>();

    for (size_t i = 0; i < keep_gids.size(0); i++) {
      BlockTypeGID gid = keep_gids_acc[i];
      new_blocks_.push_back(std::move(blocks_[index_map_[gid]]));
      new_index_map_[gid] = i;
    }

    // Save the restricted versions
    blocks_ = std::move(new_blocks_);
    index_map_ = std::move(new_index_map_);
  }

  // =================================================================
  // Public Getters / Setters
  /// @return The label of the mesh.
  const inline Label& get_label() const noexcept { return label_; }
  /// @return The number of MeshBlocks.
  inline size_t get_num_blocks() const noexcept { return blocks_.size(); }
  /// @return The number of MeshBlocks globally (across all MPI ranks).
  const inline size_t& get_global_num_blocks() const noexcept
  {
    return global_num_blocks_;
  }
  /// @return The vector of blocks.
  const inline MeshBlocks& get_blocks() const noexcept { return blocks_; }
  /// @return The global bounding box represented as a tensor with the first
  /// index of the first dimension being the minimum point in Euclidean space
  /// and the second being the maximum.
  const inline torch::Tensor& get_bbox() const noexcept { return bbox_; }
  /// @return Get the dimensionality of the blocks in the mesh
  int64_t inline get_ndim() const
  {
    is_connected_or_error("get_ndim");
    return blocks_[0]->get_ndim();
  }

  /// @param label The new label of the mesh.
  void inline set_label(const std::string& label)
  {
    label_ = Label::from_string(label);
  }
};

template<typename BlockType>
inline std::ostream& operator<<(std::ostream& os, const Mesh<BlockType>& mesh)
{
  os << "Mesh<" << BlockType::class_name
     << ">(num_blocks=" << mesh.get_num_blocks() << ")\n";

  for (const auto& bptr : mesh.get_blocks()) {
    os << " -> " << bptr->to_string() << "\n";
  }

  return os;
}

} // namespace ttnte::mesh
