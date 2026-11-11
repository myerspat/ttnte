#pragma once

#include "ttnte/utils/label.hpp"
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <string>

namespace ttnte::tensor {

/// @brief Physical meaning of a tensor index. Indices with a known dimension
/// share one label per dimension, so `Index::polar(4) == Index::polar(4)`
/// across the whole program.
enum class Dimension : uint8_t {
  POLAR,
  AZIMUTHAL,
  X,
  Y,
  Z,
  ENERGY,
  /// All spatial degrees of freedom of a patch (or of every patch) as one
  /// index.
  SPACE,
  /// A user-defined index without a physical meaning.
  OTHER,
};

/// @return The canonical name of a known dimension ("polar", "x", ...).
std::string dimension_name(Dimension dim);

/// @return Whether `dim` is one of X, Y, Z or SPACE.
inline bool is_spatial(Dimension dim) noexcept
{
  return dim == Dimension::X || dim == Dimension::Y || dim == Dimension::Z ||
         dim == Dimension::SPACE;
}

/// @brief A labeled tensor index (mode) with an optional size. A size of 0
/// marks a symbolic index whose size is bound later.
class Index {
public:
  // =================================================
  // Public types
  using Label = ttnte::utils::Label<Index>;

private:
  // =================================================
  // Private data
  int64_t size_ = 0;
  Label label_;

  Index(int64_t size, Label label) : size_(size), label_(label) {}

public:
  // =================================================
  // Public constructors
  /// @brief A symbolic index with a user label or an automatic one.
  explicit Index(std::optional<std::string> label = std::nullopt)
    : label_(label.has_value() ? Label::from_string(*label)
                               : Label::create_internal())
  {}
  /// @brief An index of a given size with a user label or an automatic one.
  Index(int64_t size, std::optional<std::string> label = std::nullopt)
    : size_(size), label_(label.has_value() ? Label::from_string(*label)
                                            : Label::create_internal())
  {}

  /// @brief The index of a known dimension.
  /// @param dim The dimension (not OTHER).
  /// @param size The size, or 0 for a symbolic index.
  static Index known(Dimension dim, int64_t size = 0);
  /// @brief The index of a known dimension, by name. Accepted names are the
  /// canonical ones ("polar", "azimuthal", "x", "y", "z", "energy", "space")
  /// and the aliases "mu", "gamma", "E" and "s".
  /// @throws std::runtime_error If the name is not a known dimension.
  static Index from_name(const std::string& name, int64_t size = 0);

  static Index polar(int64_t size = 0) { return known(Dimension::POLAR, size); }
  static Index azimuthal(int64_t size = 0)
  {
    return known(Dimension::AZIMUTHAL, size);
  }
  static Index x(int64_t size = 0) { return known(Dimension::X, size); }
  static Index y(int64_t size = 0) { return known(Dimension::Y, size); }
  static Index z(int64_t size = 0) { return known(Dimension::Z, size); }
  static Index energy(int64_t size = 0)
  {
    return known(Dimension::ENERGY, size);
  }
  static Index space(int64_t size = 0) { return known(Dimension::SPACE, size); }

  // =================================================
  // Operators
  /// @brief Indices are equal when they share a label and a size.
  friend bool operator==(const Index& A, const Index& B)
  {
    return A.label_ == B.label_ && A.size_ == B.size_;
  }
  friend bool operator!=(const Index& A, const Index& B) { return !(A == B); }
  bool operator<(const Index& other) const { return label_ < other.label_; }
  friend std::ostream& operator<<(std::ostream& os, const Index& idx);

  // =================================================
  // Public methods
  bool is_symbolic() const noexcept { return size_ == 0; }
  /// @return The same index (same label) with another size.
  Index resized(int64_t size) const { return Index(size, label_); }
  /// @return Whether both indices carry the same label, whatever their sizes.
  bool same_label(const Index& other) const noexcept
  {
    return label_ == other.label_;
  }
  /// @return The known dimension of this index, or OTHER.
  Dimension dimension() const;
  /// @return The label's name.
  std::string name() const { return label_.to_string(); }

  // =================================================
  // Getters / Setters
  int64_t size() const noexcept { return size_; }
  const Label& label() const noexcept { return label_; }
};

} // namespace ttnte::tensor

namespace std {
template<>
struct hash<ttnte::tensor::Index> {
  size_t operator()(const ttnte::tensor::Index& idx) const noexcept
  {
    return std::hash<uint64_t> {}(idx.label().to_int());
  }
};

} // namespace std
