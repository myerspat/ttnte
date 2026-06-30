#pragma once

#include "ttnte/parallel/stream_handle.hpp"
#include <c10/util/SmallVector.h>
#include <memory>
#include <optional>
#include <torch/extension.h>

namespace ttnte::parallel {

/// @brief A stream pool manager for torch CUDA streams.
class StreamPool {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<StreamPool>;

private:
  // =================================================================
  // Private data
  /// Vector of available streams.
  c10::SmallVector<StreamHandle, 16> streams_;
  /// GPU device for the streams.
  torch::Device device_;
  /// Mutex of the class for thread safety.
  std::mutex mutex_;

  // =================================================================
  // Private constructors
  StreamPool(int num_streams = 16);

public:
  ~StreamPool() = default;

  // Prevent copying
  StreamPool(const StreamPool&) = delete;
  StreamPool& operator=(const StreamPool&) = delete;

  // =================================================================
  // Public methods
  /// @return A shared pointer to the instance of the stream pool for this MPI
  /// rank.
  static Ptr instance(int num_streams = 16);
  /// @brief Try to acquire a free stream. If the optional pointer is empty then
  /// there is not an available stream.
  /// @return An optional pointer to an available stream.
  std::optional<StreamHandle> try_acquire();
  /// @brief Return the stream to the stream pool.
  /// @param stream The returning stream.
  void release(const StreamHandle& stream);

  // =================================================================
  // Public getters
  /// @return The device for these streams.
  torch::Device get_device() const noexcept { return device_; }
};

} // namespace ttnte::parallel
