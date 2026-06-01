#include "ttnte/xs/material.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::xs {

// =================================================================
// Private methods
void Material::check_finalize(const std::string& method_name) const
{
  if (is_finalized_) {
    throw utils::runtime_error(*this, error_context(method_name),
      "This material has already been finalized");
  }
}

// =================================================================
// Public methods
void Material::finalize()
{
  // Checking function
  auto check_tensor = [this](const torch::Tensor& tensor, const int64_t& ndim,
                        const std::string& method_name) {
    // Check the number of dimensions
    if (!tensor.defined() || tensor.ndimension() != ndim) {
      throw utils::runtime_error(*this, error_context(method_name),
        "Cross section tensor expected to be " + std::to_string(ndim) +
          "-dimensional");
    }

    // Check the number of groups
    if (num_groups_ == 0) {
      this->num_groups_ = tensor.size(0);
    } else if (tensor.size(0) != this->num_groups_) {
      throw utils::runtime_error(*this, method_name,
        "Cross section tensor expected to have " + std::to_string(num_groups_) +
          " groups");
    }
  };

  // Check total XS tensor
  check_tensor(total_, 1, "set_total");

  // Check the group-to-group scattering tensor
  if (!scatter_gtg_.defined() || scatter_gtg_.ndimension() != 3 ||
      scatter_gtg_.size(1) != num_groups_ ||
      scatter_gtg_.size(2) != num_groups_) {
    throw utils::runtime_error(*this, error_context("set_scatter_gtg"),
      "The group-to-group scattering tensor must be 3-dimensional of shape\n "
      "(L, G, G') for L moments and G=G' energy groups");
  }

  // Check fissile XSs
  if (is_fissile_) {
    static const std::array<std::string, 2> xs_1d_names = {"chi", "nu_fission"};
    const std::array<torch::Tensor, 2> xss = {chi_, nu_fission_};
    assert(xs_1d_names.size() == xss.size());

    // Iterate through 1-D XSs
    for (size_t i = 0; i < xs_1d_names.size(); i++) {
      check_tensor(xss[i], 1, "set_" + xs_1d_names[i]);
    }
  }

  // Array of XS names and tensors to check
  static const std::array<std::string, 3> xs_1d_names = {
    "absorption", "fission", "kappa_fission"};
  const std::array<torch::Tensor, 3> xss = {
    absorption_, fission_, kappa_fission_};
  assert(xs_1d_names.size() == xss.size());

  // Iterate through 1-D XSs
  for (size_t i = 0; i < xs_1d_names.size(); i++) {
    if (xss[i].defined()) {
      check_tensor(xss[i], 1, "set_" + xs_1d_names[i]);
    }
  }

  // Calculate absorption if not given
  if (!absorption_.defined()) {
    if (is_fissile_ && !fission_.defined()) {
      throw utils::runtime_error(*this, error_context("finalize"),
        "Either the fission or absorption cross sections must be set for a\n "
        "fissile material");
    }
    absorption_ = total_ - scatter_gtg_.sum(0);
    if (is_fissile_) {
      absorption_.sub_(fission_);
    }

    auto mask = absorption_ < 0;
    if ((absorption_ < 0).any().item<bool>()) {
      std::cout << "Warning: Negative absorption detected at indices:\n"
                << mask.nonzero() << std::endl
                << "Clamping them to 0.0\n";
      absorption_.clamp_min_(0.0);
    }
  }

  is_finalized_ = true;
}

void Material::set_chi(torch::Tensor chi)
{
  if (chi.is_nonzero()) {
    check_finalize("set_chi");
    chi_ = std::move(chi);
    is_fissile_ = true;
  }
}
void Material::set_total(torch::Tensor total)
{
  check_finalize("set_total");
  total_ = std::move(total);
}
void Material::set_absorption(torch::Tensor absorption)
{
  check_finalize("set_absorption");
  absorption_ = std::move(absorption);
}
void Material::set_kappa_fission(torch::Tensor kappa_fission)
{
  check_finalize("set_kappa_fission");
  absorption_ = std::move(kappa_fission);
}
void Material::set_fission(torch::Tensor fission)
{
  if (fission.is_nonzero()) {
    check_finalize("set_fission");
    fission_ = std::move(fission);
    is_fissile_ = true;
  }
}
void Material::set_nu_fission(torch::Tensor nu_fission)
{
  if (nu_fission.is_nonzero()) {
    check_finalize("set_nu_fission");
    nu_fission_ = std::move(nu_fission);
    is_fissile_ = true;
  }
}
void Material::set_scatter_gtg(torch::Tensor scatter_gtg)
{
  check_finalize("set_scatter_gtg");
  scatter_gtg_ = std::move(scatter_gtg);
}

} // namespace ttnte::xs
