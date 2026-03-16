#pragma once

#include "ttnte/utils/label.hpp"
#include <optional>
#include <string>

namespace ttnte::cad {

template<typename T>
class Basis {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<Basis>;

protected:
  // =================================================================
  // Protected data
  Label label_;

  // =================================================================
  // Protected methods
  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return std::string("ttnte::cad::") + T::class_name + "::" + func_name;
  }

public:
  // =================================================================
  // Public constructors
  Basis(std::optional<std::string> label)
    : label_(label.has_value() ? Label::from_string(*label)
                               : Label::create_internal())
  {}
  Basis(const Label& label) : label_(label) {}

  // =================================================================
  // Public Getters / Setters
  const Label& get_label() const noexcept { return label_; }
};

} // namespace ttnte::cad
