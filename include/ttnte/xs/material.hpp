#pragma once

#include "ttnte/utils/label.hpp"
#include <optional>
#include <torch/extension.h>

namespace ttnte::xs {

class Material {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<Material>;

private:
  // =================================================================
  // Private data
  Label label_;
  int64_t num_groups_ = 0;
  bool is_finalized_ = false;
  bool is_fissile_ = false;

  // XS information
  torch::Tensor chi_;
  torch::Tensor total_;
  torch::Tensor absorption_;
  torch::Tensor kappa_fission_;
  torch::Tensor fission_;
  torch::Tensor nu_fission_;
  torch::Tensor scatter_gtg_;

  // =================================================================
  // Private methods
  void check_finalize(const std::string& method_name) const;

  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return std::string("ttnte::xs::Material::") + func_name;
  }

public:
  // =================================================================
  // Public constructors
  Material(std::optional<Label> label = std::nullopt)
    : label_(label.value_or(Label::create_internal()))
  {}
  Material(const std::string& label) : label_(Label::from_string(label)) {}

  // =================================================================
  // Public methods
  bool is_finalized() const noexcept { return is_finalized_; }
  bool is_fissile() const noexcept { return is_fissile(); }
  void finalize();

  // =================================================================
  // Public Getters / Setters
  const Label& get_label() const noexcept { return label_; }
  const int64_t& get_num_groups() const noexcept { return num_groups_; }
  int64_t get_num_moments() const noexcept
  {
    return scatter_gtg_.defined() ? scatter_gtg_.size(0) : 0;
  }

  const torch::Tensor& get_chi() const noexcept { return chi_; }
  const torch::Tensor& get_total() const noexcept { return total_; }
  const torch::Tensor& get_absorption() const noexcept { return absorption_; }
  const torch::Tensor& get_kappa_fission() const noexcept
  {
    return kappa_fission_;
  }
  const torch::Tensor& get_fission() const noexcept { return fission_; }
  const torch::Tensor& get_nu_fission() const noexcept { return nu_fission_; }
  const torch::Tensor& get_scatter_gtg() const noexcept { return scatter_gtg_; }

  void set_chi(torch::Tensor chi);
  void set_total(torch::Tensor total);
  void set_absorption(torch::Tensor absorption);
  void set_kappa_fission(torch::Tensor kappa_fission);
  void set_fission(torch::Tensor fission);
  void set_nu_fission(torch::Tensor nu_fission);
  void set_scatter_gtg(torch::Tensor scatter_gtg);
};

} // namespace ttnte::xs
