#pragma once

#include "ttnte/tensor/index.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ttnte::tensor {

/// @brief The order of the indices of a tensor train and their grouping into
/// cores. Core k holds the indices `cores()[k]`, merged in the given order.
///
/// A layout is built from a string, cores separated by '|' and the indices of
/// a core by ',' (e.g. "polar|azimuthal|x,y|energy"), or from lists of
/// Index. Spatial indices are either the single index `space` or the
/// Cartesian indices x (, y, z); they must be contiguous so that the solver
/// can treat all of space as one block.
class Layout {
public:
  // =================================================
  // Public types
  using Core = std::vector<Index>;

private:
  // =================================================
  // Private data
  std::vector<Core> cores_;

public:
  // =================================================
  // Public constructors
  Layout() = default;
  /// @throws std::runtime_error If a core is empty or an index repeats.
  explicit Layout(std::vector<Core> cores);
  /// @brief Parse a layout string.
  /// @throws std::runtime_error On an unknown index name or a malformed
  /// string.
  explicit Layout(const std::string& spec);

  /// @brief ttnte's default order: (polar[, azimuthal], x[, y, z], energy),
  /// one index per core, or with all of space in one `space` core.
  static Layout canonical(int64_t num_dim, bool split_space = true);

  // =================================================
  // Public methods
  /// @brief Check that the layout describes an angular flux in `num_dim`
  /// spatial dimensions: polar and energy present, azimuthal iff
  /// num_dim >= 2, and either `space` or exactly the first num_dim of
  /// x, y, z, contiguous.
  /// @throws std::runtime_error If it does not.
  void validate(int64_t num_dim) const;
  /// @return The layout with every spatial index merged into one `space`
  /// index, in place of the first spatial index (the layout the solver
  /// samples in).
  Layout merge_space() const;
  /// @return The layout with every index resized from `sizes` (indices not
  /// in the map keep their size).
  Layout with_sizes(const std::map<Dimension, int64_t>& sizes) const;
  /// @return The core holding `dim`, or -1.
  int64_t core_of(Dimension dim) const;
  /// @return Whether any index has the dimension `dim`.
  bool contains(Dimension dim) const { return core_of(dim) >= 0; }
  /// @return The product of a core's index sizes.
  int64_t core_size(size_t k) const;
  /// @return The indices of every core, in order.
  std::vector<Index> indices() const;
  /// @return The layout string ("polar|azimuthal|space|energy").
  std::string to_string() const;

  friend bool operator==(const Layout& A, const Layout& B);
  friend bool operator!=(const Layout& A, const Layout& B) { return !(A == B); }

  // =================================================
  // Getters / Setters
  const std::vector<Core>& cores() const noexcept { return cores_; }
  const Core& core(size_t k) const { return cores_.at(k); }
  size_t num_cores() const noexcept { return cores_.size(); }
};

} // namespace ttnte::tensor
