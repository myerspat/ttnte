#pragma once

#include "ttnte/physics/boundary_types.hpp"
#include "ttnte/utils/io_formatting.hpp"
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <torch/extension.h>

namespace ttnte::mesh {

/// @brief Key for hashing based on a point in space.
class PointKey {
private:
  // =================================================================
  // Private data
  /// The point in physical space (1-D tensor).
  torch::Tensor point_;
  /// The tolerance to apply for rounding.
  double tol_;

public:
  // =================================================================
  // Public constructors
  PointKey(const torch::Tensor& point, const double& tol)
  {
    TORCH_CHECK(point.ndimension() == 1, "The point must be 1-dimensional");
    point_ = point;
    tol_ = tol;
  }

  // =================================================================
  // Public operators
  bool operator==(const PointKey& other) const
  {
    TORCH_CHECK(point_.size(0) == other.point_.size(0));
    return torch::allclose(point_, other.point_, tol_, tol_);
  }

  // =================================================================
  // Public getters
  const torch::Tensor& get_point() const noexcept { return point_; }
};

/// @brief Hash for the PointKey.
struct PointHash {
  // =================================================================
  // Public operators
  std::size_t operator()(const PointKey& k) const
  {
    std::size_t seed = 0;
    const auto& point = k.get_point();

    for (size_t i = 0; i < point.size(0); i++) {
      seed ^= std::hash<double> {}(point[i].item<double>()) + 0x9e3779b9 +
              (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

/// @brief Mapping information for receiving information from a
/// neighboring MeshBlock. Note that we apply the permutation first
/// and then the flip.
struct BoundaryMapping {
  /// Flip the direction along each axis for which we sweep across
  c10::SmallVector<bool, 2> flip;
  /// Axes permutation
  c10::SmallVector<int64_t, 3> perm;
};

/// @brief The connected MeshBlock information.
struct NeighborInfo {
  /// Global (same across MPI ranks) ID
  int64_t gid;
  /// Index into the boundary vector for each MeshBlock
  size_t fid;
  /// The rank that owns this MeshBlock
  int mpi_rank;
  /// Mapping that must be applied to any data passed from
  /// this neighbor
  BoundaryMapping mapping;

  /// @return To string method for printing
  std::string to_string() const
  {
    std::stringstream ss;
    ss << "BoundaryMapping(gid=" << gid << ", fid=" << fid
       << ", mpi_rank=" << mpi_rank << ")";
    return ss.str();
  }
};

class BoundaryInfo {
private:
  // =================================================================
  // Private data
  /// Face ID where (dim, is_upper) is mapped to a number
  int64_t fid_ = -1;

  /// Boundary type
  physics::BoundaryType type_ = physics::BoundaryType::UNKNOWN;

  /// If type == ttnte::physics::BoundaryType::INTERNAL this is the connected
  /// mesh block boundary
  c10::SmallVector<NeighborInfo, 2> connections_;

public:
  // =================================================================
  // Public constructors
  BoundaryInfo(size_t dim, bool is_upper)
  {
    // Map the dimension and is_upper to an ID
    fid_ = dim * 2 + static_cast<size_t>(is_upper);
  }

  // =================================================================
  // Public methods
  void add_connection(const NeighborInfo& ninfo)
  {
    connections_.push_back(ninfo);
  }

  // =================================================================
  // Public getters
  const int64_t& get_fid() const noexcept { return fid_; }
  const physics::BoundaryType& get_type() const noexcept { return type_; }
  const c10::SmallVector<NeighborInfo, 2>& get_connections() const noexcept
  {
    return connections_;
  }

  void set_type(const physics::BoundaryType& type) { type_ = type; }
};

inline std::ostream& operator<<(std::ostream& os, const BoundaryMapping& bm)
{
  os << "BoundaryMapping(flip=(";

  std::string sep = "";
  for (bool b : bm.flip) {
    os << sep << (b ? "True" : "False");
    sep = ", ";
  }

  os << "), perm=(";
  sep = "";
  for (const auto& p : bm.perm) {
    os << sep << p;
    sep = ", ";
  }
  os << "))";

  return os;
}

inline std::ostream& operator<<(std::ostream& os, const NeighborInfo& ninfo)
{
  os << "NeighborInfo(\n  gid=" << ninfo.gid << ",\n  fid=" << ninfo.fid
     << ",\n  mpi_rank=" << ninfo.mpi_rank << ",\n  mapping=" << ninfo.mapping
     << "\n)";

  return os;
}

inline std::ostream& operator<<(std::ostream& os, const BoundaryInfo& binfo)
{

  os << "BoundaryInfo(\n  fid=" << binfo.get_fid()
     << ",\n  type=" << physics::to_string(binfo.get_type())
     << ",\n  connections=[";

  if (binfo.get_connections().empty()) {
    os << "],\n)";
    return os;
  } else {
    os << "\n";
  }

  std::stringstream ss;
  for (const auto& connection : binfo.get_connections()) {
    ss << connection.to_string() << ",\n";
  }
  os << utils::indent_message(ss.str(), 4) << "  ],\n)";

  return os;
}

} // namespace ttnte::mesh
