#pragma once

#include "ttnte/utils/exception.hpp"
#include <cassert>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ttnte::utils {

template<typename T>
class SymbolTable {
private:
  std::unordered_map<uint64_t, std::string> map_;
  std::unordered_map<std::string, int64_t> versions_;
  mutable std::mutex mutex_;

  // Private Constructor
  SymbolTable() = default;

  // =================================================
  // Private methods
  void register_symbol(uint64_t id, const std::string& name)
  {
    // This symbol does not exist
    auto it = map_.find(id);
    if (it != map_.end()) {
      // If the ID exists but the name is DIFFERENT, we have a hash collision!
      if (it->second != name) {
        std::cerr << "[WARNING] Hash collision in SymbolTable! "
                  << "ID " << id << " maps to both '" << it->second << "' and '"
                  << name << "'" << std::endl;
      }
      return; // Already registered
    }

    map_.insert({id, name});
  }

public:
  // Delete copy/move for Singleton safety
  SymbolTable(const SymbolTable&) = delete;
  SymbolTable& operator=(const SymbolTable&) = delete;
  SymbolTable(SymbolTable&&) = delete;
  SymbolTable& operator=(SymbolTable&&) = delete;

  // =================================================
  // Public methods
  std::string lookup(uint64_t id) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(id);
    if (it != map_.end()) {
      return it->second;
    }
    return ""; // Unknown
  }

  int64_t increment_version(const std::string& name)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (versions_.contains(name)) {
      auto& version = versions_[name];
      version += 1;
      return version;
    }

    throw utils::runtime_error("ttnte::utils::SymbolTable::increment_version",
      "The version map does not contain this name " + name);
  }

  // =================================================
  // Getters / Setters
  // Singleton Access
  static SymbolTable& instance()
  {
    static SymbolTable instance;
    return instance;
  }

  // Register ID and its string name
  void register_new(uint64_t id, const std::string& name)
  {
    register_symbol(id, name);
    assert(!versions_.contain(name));
    versions_.insert({name, 0});
  }

  void register_clone(uint64_t id, const std::string& name)
  {
    register_symbol(id, name);
  }
};

} // namespace ttnte::utils
