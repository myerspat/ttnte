#pragma once

#include "ttnte/utils/label.hpp"
#include "ttnte/xs/material.hpp"
#include <memory>
#include <optional>

namespace ttnte::xs {

class Server {
public:
  // =================================================================
  // Public types
  using Label = utils::Label<Server>;
  using Ptr = std::shared_ptr<Server>;
  using MaterialIDs = c10::SmallVector<Material::Label::ID, 10>;
  using MaterialMap = std::unordered_map<Material::Label::ID, Material>;

private:
  // =================================================================
  // Private data
  Label label_;
  int64_t num_groups_ = 0;
  int64_t num_moments_ = 0;
  MaterialIDs mat_ids_;
  MaterialMap mat_map_;
  bool is_finalized_ = false;

  // =================================================================
  // Private methods
  void check_finalize(const std::string& method_name) const;

  [[nodiscard]] std::string error_context(const std::string& func_name) const
  {
    return std::string("ttnte::xs::Server::") + func_name;
  }

  // =================================================================
  // Private constructors
  Server(std::optional<Label> label = std::nullopt)
    : label_(label.value_or(Label::create_internal()))
  {}
  Server(const std::string& label) : label_(Label::from_string(label)) {}

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return std::shared_ptr<Server>(new Server(std::forward<Args>(args)...));
  }

  bool is_finalized() const noexcept { return is_finalized_; }
  void finalize();
  void add_material(Material mat);

  // =================================================================
  // Public Getters / Setters
  const Label& get_label() const noexcept { return label_; }
  const int64_t& get_num_groups() const noexcept { return num_groups_; }
  const int64_t& get_num_moments() const noexcept { return num_moments_; }
  const MaterialIDs& get_material_ids() const noexcept { return mat_ids_; }
  const MaterialMap& get_material_map() const noexcept { return mat_map_; }
  const Material& get_material(const Material::Label::ID& mat_id) const;
  const Material& get_material(const Material::Label& mat_label) const;
  int64_t get_num_materials() const noexcept { return mat_ids_.size(); }
};

} // namespace ttnte::xs
