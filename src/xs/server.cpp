#include "ttnte/xs/server.hpp"
#include "ttnte/utils/exception.hpp"
#include "ttnte/xs/material.hpp"
#include <cmath>
#include <sstream>

namespace ttnte::xs {

// =================================================================
// Private methods
void Server::check_finalize(const std::string& method_name) const
{
  if (is_finalized_) {
    throw utils::runtime_error(*this, error_context(method_name),
      "This material has already been finalized");
  }
}

// =================================================================
// Public methods
void Server::finalize()
{
  // Run checks on material IDs and material map
  is_finalized_ = true;
}

void Server::set_group_bounds(std::vector<double> bounds)
{
  if (bounds.empty()) {
    group_bounds_.clear();
    return;
  }
  const auto where = error_context("set_group_bounds");
  if (bounds.size() < 2)
    throw utils::runtime_error(*this, where, "Need at least two boundaries");
  if (num_groups_ > 0 &&
      static_cast<int64_t>(bounds.size()) != num_groups_ + 1) {
    throw utils::runtime_error(*this, where,
      "Expected " + std::to_string(num_groups_ + 1) + " boundaries for " +
        std::to_string(num_groups_) + " groups, got " +
        std::to_string(bounds.size()));
  }
  for (size_t i = 0; i < bounds.size(); i++) {
    if (!(bounds[i] > 0.0)) {
      throw utils::runtime_error(*this, where,
        "Boundaries must be positive (a lethargy width ln(E_{g-1} / E_g) needs "
        "E_g > 0): give the lowest edge as a small cutoff such as 1e-5 eV");
    }
    if (i > 0 && !(bounds[i] < bounds[i - 1])) {
      throw utils::runtime_error(*this, where,
        "Boundaries must be strictly decreasing (group 0 fastest)");
    }
  }
  group_bounds_ = std::move(bounds);
}

std::vector<double> Server::lethargy_widths() const
{
  std::vector<double> du;
  for (size_t g = 0; g + 1 < group_bounds_.size(); g++)
    du.push_back(std::log(group_bounds_[g] / group_bounds_[g + 1]));
  return du;
}

void Server::add_material(Material mat)
{
  // Check the server isn't already finalized
  check_finalize("add_material");

  // Finalize the material
  if (!mat.is_finalized()) {
    mat.finalize();
  }

  // Check this new material has the correct number of groups and moments
  if (num_groups_ == 0) {
    num_groups_ = mat.get_num_groups();
    num_moments_ = mat.get_num_moments();
    if (!group_bounds_.empty() &&
        static_cast<int64_t>(group_bounds_.size()) != num_groups_ + 1) {
      throw utils::runtime_error(*this, error_context("add_material"),
        "The group boundaries do not match the material's number of groups");
    }
  }
  if (num_groups_ != mat.get_num_groups() ||
      num_moments_ != mat.get_num_moments()) {
    std::stringstream ss;
    ss << "Material with label " << mat.get_label()
       << " has a different number of moments or groups\n"
          " than the other materials";
    throw utils::runtime_error(*this, error_context("add_material"), ss.str());
  }

  // Add material to the map
  Material::Label::ID mat_id = mat.get_label().to_int();
  if (!mat_map_.contains(mat_id)) {
    mat_map_[mat_id] = mat;
    mat_ids_.push_back(mat_id);

  } else {
    std::stringstream ss;
    ss << "Material with label " << mat.get_label() << " already exists";
    throw utils::runtime_error(*this, error_context("add_material"), ss.str());
  }
}

const Material& Server::get_material(const Material::Label::ID& mat_id) const
{
  if (mat_map_.contains(mat_id)) {
    return mat_map_.at(mat_id);
  }
  throw utils::runtime_error(*this, error_context("get_material"),
    "Material ID " + std::to_string(mat_id) + " does not exist");
}

const Material& Server::get_material(const Material::Label& mat_label) const
{
  return get_material(mat_label.to_int());
}

} // namespace ttnte::xs
