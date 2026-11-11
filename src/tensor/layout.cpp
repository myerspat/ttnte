#include "ttnte/tensor/layout.hpp"
#include "ttnte/utils/exception.hpp"
#include <sstream>

namespace ttnte::tensor {

namespace {

std::string trim(const std::string& s)
{
  const auto first = s.find_first_not_of(" \t");
  if (first == std::string::npos)
    return "";
  const auto last = s.find_last_not_of(" \t");
  return s.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string& s, char delim)
{
  std::vector<std::string> parts;
  std::stringstream ss(s);
  std::string part;
  while (std::getline(ss, part, delim))
    parts.push_back(trim(part));
  if (!s.empty() && s.back() == delim)
    parts.push_back("");
  return parts;
}

} // namespace

Layout::Layout(std::vector<Core> cores) : cores_(std::move(cores))
{
  std::vector<Index> seen;
  for (const auto& core : cores_) {
    if (core.empty()) {
      throw utils::runtime_error(
        "ttnte::tensor::Layout::Layout", "A core holds no index");
    }
    for (const auto& idx : core) {
      for (const auto& other : seen) {
        if (other.same_label(idx)) {
          throw utils::runtime_error("ttnte::tensor::Layout::Layout",
            "Index '" + idx.name() + "' appears more than once");
        }
      }
      seen.push_back(idx);
    }
  }
}

Layout::Layout(const std::string& spec)
  : Layout([&] {
      std::vector<Core> cores;
      for (const auto& core_spec : split(spec, '|')) {
        Core core;
        for (const auto& name : split(core_spec, ',')) {
          if (name.empty()) {
            throw utils::runtime_error("ttnte::tensor::Layout::Layout",
              "Empty index name in layout '" + spec + "'");
          }
          core.push_back(Index::from_name(name));
        }
        cores.push_back(std::move(core));
      }
      return cores;
    }())
{}

Layout Layout::canonical(int64_t num_dim, bool split_space)
{
  std::vector<Core> cores {{Index::polar()}};
  if (num_dim >= 2)
    cores.push_back({Index::azimuthal()});
  if (split_space) {
    const Dimension xyz[] = {Dimension::X, Dimension::Y, Dimension::Z};
    for (int64_t i = 0; i < num_dim; i++)
      cores.push_back({Index::known(xyz[i])});
  } else {
    cores.push_back({Index::space()});
  }
  cores.push_back({Index::energy()});
  return Layout(std::move(cores));
}

void Layout::validate(int64_t num_dim) const
{
  const std::string ctx = "ttnte::tensor::Layout::validate";
  if (num_dim < 1 || num_dim > 3)
    throw utils::runtime_error(ctx, "num_dim must be 1, 2 or 3");
  const auto all = indices();
  for (const auto& idx : all) {
    if (idx.dimension() == Dimension::OTHER) {
      throw utils::runtime_error(
        ctx, "Index '" + idx.name() + "' is not an angular flux index");
    }
  }
  auto require = [&](Dimension dim, bool present) {
    if (contains(dim) != present) {
      throw utils::runtime_error(ctx, "Layout '" + to_string() + "' must " +
                                        (present ? "" : "not ") + "contain '" +
                                        dimension_name(dim) + "' in " +
                                        std::to_string(num_dim) + "-D");
    }
  };
  require(Dimension::POLAR, true);
  require(Dimension::AZIMUTHAL, num_dim >= 2);
  require(Dimension::ENERGY, true);

  const bool merged = contains(Dimension::SPACE);
  const Dimension xyz[] = {Dimension::X, Dimension::Y, Dimension::Z};
  for (int64_t i = 0; i < 3; i++) {
    if (merged || i >= num_dim)
      require(xyz[i], false);
    else
      require(xyz[i], true);
  }

  // Spatial indices must be contiguous in the flattened order
  std::vector<size_t> positions;
  for (size_t k = 0; k < all.size(); k++) {
    if (is_spatial(all[k].dimension()))
      positions.push_back(k);
  }
  if (positions.back() - positions.front() + 1 != positions.size()) {
    throw utils::runtime_error(
      ctx, "The spatial indices of '" + to_string() + "' must be contiguous");
  }
}

Layout Layout::merge_space() const
{
  std::vector<Core> cores;
  int64_t space_size = 1;
  bool any_symbolic = false;
  for (const auto& idx : indices()) {
    if (is_spatial(idx.dimension())) {
      space_size *= idx.size();
      any_symbolic = any_symbolic || idx.is_symbolic();
    }
  }
  bool placed = false;
  for (const auto& core : cores_) {
    Core merged;
    for (const auto& idx : core) {
      if (!is_spatial(idx.dimension())) {
        merged.push_back(idx);
      } else if (!placed) {
        merged.push_back(Index::space(any_symbolic ? 0 : space_size));
        placed = true;
      }
    }
    if (!merged.empty())
      cores.push_back(std::move(merged));
  }
  return Layout(std::move(cores));
}

Layout Layout::with_sizes(const std::map<Dimension, int64_t>& sizes) const
{
  std::vector<Core> cores = cores_;
  for (auto& core : cores) {
    for (auto& idx : core) {
      auto it = sizes.find(idx.dimension());
      if (it != sizes.end())
        idx = idx.resized(it->second);
    }
  }
  return Layout(std::move(cores));
}

int64_t Layout::core_of(Dimension dim) const
{
  for (size_t k = 0; k < cores_.size(); k++) {
    for (const auto& idx : cores_[k]) {
      if (idx.dimension() == dim)
        return static_cast<int64_t>(k);
    }
  }
  return -1;
}

int64_t Layout::core_size(size_t k) const
{
  int64_t size = 1;
  for (const auto& idx : cores_.at(k))
    size *= idx.size();
  return size;
}

std::vector<Index> Layout::indices() const
{
  std::vector<Index> all;
  for (const auto& core : cores_)
    all.insert(all.end(), core.begin(), core.end());
  return all;
}

std::string Layout::to_string() const
{
  std::string s;
  for (size_t k = 0; k < cores_.size(); k++) {
    if (k > 0)
      s += "|";
    for (size_t i = 0; i < cores_[k].size(); i++) {
      if (i > 0)
        s += ",";
      s += cores_[k][i].name();
    }
  }
  return s;
}

bool operator==(const Layout& A, const Layout& B)
{
  return A.cores_ == B.cores_;
}

} // namespace ttnte::tensor
