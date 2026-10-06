#pragma once

#include "ttnte/physics/boundary_types.hpp"
#include "ttnte/physics/fixed_source.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/utils/io_formatting.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
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

public:
  // =================================================================
  // Public constructors
  PointKey(const torch::Tensor& point, const double& tol)
  {
    TORCH_CHECK(point.ndimension() == 1, "The point must be 1-dimensional");
    point_ = torch::round(point / tol) * tol;
  }

  // =================================================================
  // Public operators
  bool operator==(const PointKey& other) const
  {
    TORCH_CHECK(point_.size(0) == other.point_.size(0));
    return torch::equal(point_, other.point_);
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
  /// Index into the boundary vector for each MeshBlock (opaque face ID used
  /// for MPI tag generation; do not decompose arithmetically)
  size_t fid;
  /// The rank that owns this MeshBlock
  int mpi_rank;
  /// Physical dimension of the SOURCE's (neighbor's) face
  size_t dim = 0;
  /// Orientation of the SOURCE's (neighbor's) face
  bool is_upper = false;
  /// Mapping that must be applied to any data passed from
  /// this neighbor
  BoundaryMapping mapping;

  /// @return To string method for printing
  std::string to_string() const
  {
    std::stringstream ss;
    ss << "NeighborInfo(gid=" << gid << ", fid=" << fid
       << ", mpi_rank=" << mpi_rank << ", dim=" << dim
       << ", is_upper=" << (is_upper ? "true" : "false") << ")";
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

  /// If type == ttnte::physics::BoundaryType::INCIDENT this is the
  /// prescribed incident flux specification for this face.
  std::optional<physics::FixedSource> source_ = std::nullopt;

  /// Specular albedo of the face, in [0, 1]: the incoming angular flux is
  /// albedo * (mirrored outgoing angular flux). Only meaningful when
  /// type == ttnte::physics::BoundaryType::REFLECTIVE (1 = perfect specular
  /// reflection, 0 = vacuum); the assembler rejects albedo != 1 on any other
  /// face type.
  double albedo_ = 1.0;

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
  c10::SmallVector<NeighborInfo, 2>& get_connections() { return connections_; }
  /// @return The prescribed incident source for this face, if one was set.
  const std::optional<physics::FixedSource>& get_source() const noexcept
  {
    return source_;
  }
  /// @return The specular albedo of this face (1 unless set).
  double albedo() const noexcept { return albedo_; }

  void set_type(const physics::BoundaryType& type) { type_ = type; }
  /// @brief Set the specular albedo of this face.
  /// @param albedo The fraction of the mirrored outgoing angular flux that
  /// re-enters through this face, in [0, 1].
  /// @throws utils::runtime_error If albedo is outside [0, 1] (or NaN).
  void set_albedo(double albedo)
  {
    if (!(albedo >= 0.0 && albedo <= 1.0)) {
      throw utils::runtime_error("ttnte::mesh::BoundaryInfo::set_albedo",
        "The albedo must be in [0, 1], got " + std::to_string(albedo));
    }
    albedo_ = albedo;
  }
  /// @param source The prescribed incident source for this face.
  void set_source(physics::FixedSource source) { source_ = std::move(source); }
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
     << ",\n  mpi_rank=" << ninfo.mpi_rank << ",\n  dim=" << ninfo.dim
     << ",\n  is_upper=" << (ninfo.is_upper ? "true" : "false")
     << ",\n  mapping=" << ninfo.mapping << "\n)";

  return os;
}

inline std::ostream& operator<<(std::ostream& os, const BoundaryInfo& binfo)
{

  os << "BoundaryInfo(\n  fid=" << binfo.get_fid()
     << ",\n  type=" << physics::to_string(binfo.get_type())
     << ",\n  albedo=" << binfo.albedo() << ",\n  connections=[";

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
