#pragma once

#include "ttnte/utils/parallel_context.hpp"
#include "ttnte/utils/symbol_table.hpp"
#include <atomic>
#include <cstdint>
#include <string>

namespace ttnte::utils {

template<typename T>
class Label {
private:
  uint64_t id_;

  // --- Bit Layout (64-bit) ---
  // [63]    User Flag (1=String, 0=Auto)
  // [48-62] MPI Rank (15 bits) - Supports 32k ranks
  // [0-47]  Value (48 bits) - Counter or Hash

  static constexpr uint64_t USER_BIT = 1ULL << 63;
  static constexpr uint64_t RANK_MASK = 0x7FFFULL << 48;
  static constexpr uint64_t RANK_SHIFT = 48;
  static constexpr uint64_t VAL_MASK = 0xFFFFFFFFFFFFULL;

  // --- Static Counter (Per Type T) ---
  // C++17 inline static allows this to be defined in header safely
  inline static std::atomic<uint64_t> s_counter {1};

public:
  Label() : id_(0) {}
  explicit Label(uint64_t val) : id_(val) {}

  // =================================================
  // Auto-Generator
  // =================================================
  static Label create_internal()
  {
    // 1. Get MPI Rank
    uint64_t rank =
      static_cast<uint64_t>(utils::ParallelContext::instance().rank()) & 0x7FFF;

    // 2. Increment Type-Specific Counter
    uint64_t count = s_counter.fetch_add(1);

    // 3. Assemble: [USER=0 | RANK | COUNTER]
    return Label((rank << RANK_SHIFT) | (count & VAL_MASK));
  }

  // =================================================
  // User String Factory
  // =================================================
  static Label from_string(const std::string& name, bool is_clone = false)
  {
    uint64_t hash = 0xcbf29ce484222325;
    for (char c : name) {
      hash ^= static_cast<uint64_t>(c);
      hash *= 0x100000001b3;
    }

    // 1. Set User Bit
    // 2. Mask Hash to 48 bits
    uint64_t id = USER_BIT | (hash & VAL_MASK);

    // 3. Register in Type-Specific Symbol Table
    if (!is_clone) {
      utils::SymbolTable<T>::instance().register_new(id, name);
    } else {
      utils::SymbolTable<T>::instance().register_clone(id, name);
    }

    return Label(id);
  }

  // =================================================
  // User String Factory
  // =================================================
  Label clone() const
  {
    if (!is_valid())
      return Label();
    if (!is_user_defined())
      return create_internal();

    // Look up the name and add clone string
    std::string name = SymbolTable<T>::instance().lookup(id_);
    assert(name != "");

    // Find last "_c"
    size_t last_c = name.find_last_of("_c");

    if (last_c != std::string::npos && last_c > 0) {
      // Check if the characters following "_c" are all digits
      bool all_digits = true;
      for (size_t i = last_c + 1; i < name.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
          all_digits = false;
          break;
        }
      }

      // Find the base and increment the version
      if (all_digits && (last_c + 1 < name.size())) {
        std::string base = name.substr(0, last_c);
        return from_string(
          base + "_c" +
            std::to_string(SymbolTable<T>::instance().increment_version(base)),
          true);
      }
    }

    // First clone of this string
    return from_string(
      name + "_c" +
        std::to_string(SymbolTable<T>::instance().increment_version(name)),
      true);
  }

  // =================================================
  // 3. Inspectors & Debugging
  // =================================================
  bool is_valid() const { return id_ != 0; }
  bool is_user_defined() const { return (id_ & USER_BIT) != 0; }
  uint64_t raw() const { return id_; }

  std::string to_string() const
  {
    if (!is_valid())
      return "Null";

    // If User String, look it up in THIS type's table
    if (is_user_defined()) {
      std::string name = utils::SymbolTable<T>::instance().lookup(id_);
      if (!name.empty())
        return name;
      return "Unknown(" + std::to_string(id_ & VAL_MASK) + ")";
    }

    // If Auto, format as Rank:Count
    uint64_t rank = (id_ & RANK_MASK) >> RANK_SHIFT;
    uint64_t count = (id_ & VAL_MASK);
    return std::to_string(rank) + ":" + std::to_string(count);
  }

  // Equality
  bool operator==(const Label& other) const { return id_ == other.id_; }
  bool operator!=(const Label& other) const { return id_ != other.id_; }
  bool operator<(const Label& other) const { return id_ < other.id_; }

  friend std::ostream& operator<<(std::ostream& os, const Label& label)
  {
    os << "'" << label.to_string() << "'";
    return os;
  }
};

} // namespace ttnte::utils

namespace std {
template<typename T>
struct hash<ttnte::utils::Label<T>> {
  size_t operator()(const ttnte::utils::Label<T>& l) const
  {
    // Use the built-in hash for the underlying unique integer
    return std::hash<uint64_t> {}(l.raw());
  }
};
} // namespace std
