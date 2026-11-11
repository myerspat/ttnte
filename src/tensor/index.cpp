#include "ttnte/tensor/index.hpp"
#include "ttnte/utils/exception.hpp"
#include <array>

namespace ttnte::tensor {

namespace {

constexpr size_t NUM_KNOWN = static_cast<size_t>(Dimension::OTHER);

/// One label per known dimension, registered once (Label::from_string must
/// not register the same name twice).
const std::array<Index::Label, NUM_KNOWN>& known_labels()
{
  static const std::array<Index::Label, NUM_KNOWN> labels = [] {
    std::array<Index::Label, NUM_KNOWN> l;
    for (size_t k = 0; k < NUM_KNOWN; k++)
      l[k] =
        Index::Label::from_string(dimension_name(static_cast<Dimension>(k)));
    return l;
  }();
  return labels;
}

} // namespace

std::string dimension_name(Dimension dim)
{
  switch (dim) {
  case Dimension::POLAR:
    return "polar";
  case Dimension::AZIMUTHAL:
    return "azimuthal";
  case Dimension::X:
    return "x";
  case Dimension::Y:
    return "y";
  case Dimension::Z:
    return "z";
  case Dimension::ENERGY:
    return "energy";
  case Dimension::SPACE:
    return "space";
  default:
    return "other";
  }
}

Index Index::known(Dimension dim, int64_t size)
{
  if (dim == Dimension::OTHER) {
    throw utils::runtime_error(
      "ttnte::tensor::Index::known", "OTHER is not a known dimension");
  }
  if (size < 0) {
    throw utils::runtime_error(
      "ttnte::tensor::Index::known", "The size must be non-negative");
  }
  return Index(size, known_labels()[static_cast<size_t>(dim)]);
}

Index Index::from_name(const std::string& name, int64_t size)
{
  static const std::array<std::pair<const char*, Dimension>, 11> names {{
    {"polar", Dimension::POLAR},
    {"mu", Dimension::POLAR},
    {"azimuthal", Dimension::AZIMUTHAL},
    {"gamma", Dimension::AZIMUTHAL},
    {"x", Dimension::X},
    {"y", Dimension::Y},
    {"z", Dimension::Z},
    {"energy", Dimension::ENERGY},
    {"E", Dimension::ENERGY},
    {"space", Dimension::SPACE},
    {"s", Dimension::SPACE},
  }};
  for (const auto& [n, dim] : names) {
    if (name == n)
      return known(dim, size);
  }
  throw utils::runtime_error("ttnte::tensor::Index::from_name",
    "Unknown index name '" + name +
      "' (known: polar/mu, azimuthal/gamma, x, y, z, energy/E, space/s)");
}

Dimension Index::dimension() const
{
  const auto& labels = known_labels();
  for (size_t k = 0; k < NUM_KNOWN; k++) {
    if (labels[k] == label_)
      return static_cast<Dimension>(k);
  }
  return Dimension::OTHER;
}

std::ostream& operator<<(std::ostream& os, const Index& idx)
{
  os << "Index(" << idx.name();
  if (!idx.is_symbolic())
    os << ", " << idx.size();
  return os << ")";
}

} // namespace ttnte::tensor
