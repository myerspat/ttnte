#pragma once

#include <vector>

namespace ttnte::parallel {

/// @brief This is an interface class to MPI_Request.
class Request {
private:
  // =================================================================
  // Private data
  /// Type-erased MPI_Request
  int f_req_;

public:
  // =================================================================
  // Public constructors
  // Default constructor creates an invalid/null request
  Request() : f_req_(0) {}
  explicit Request(int f_req) : f_req_(f_req) {}

  // =================================================================
  // Public methods
  /// @brief Wait for this MPI request to finish.
  void wait();
  /// @brief Wait for a vector of MPI requests to finish.
  static void wait_all(std::vector<Request>& requests);
  /// @brief Non-blocking check to see if the request is finished.
  /// @return True if completed, false if still pending.
  bool test();
  /// @brief Non-blocking check for a vector of requests.
  /// @param requests The list of requests to check.
  /// @param completed_indices Output vector populated with the indices of
  /// completed requests.
  static void test_some(
    std::vector<Request>& requests, std::vector<int>& completed_indices);

  // =================================================================
  // Public getters / setters
  /// @return Get the type-erased MPI request.
  int get() const { return f_req_; }
  /// @param f_req The new type-erased MPI request.
  void update(int f_req) { f_req_ = f_req; }
};

} // namespace ttnte::parallel
