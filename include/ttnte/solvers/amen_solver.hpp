#pragma once

#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::solvers {

/// @brief The AMEn local solver which calls torchTT's implementation.
class AMEnSolver : public LocalSolver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<AMEnSolver>;

protected:
  // =================================================================
  // Protected data
  /// Number of sweeps
  int nswp_;
  /// Relative residual.
  double eps_;
  /// Maximum allowed rank.
  int max_rank_;
  /// The largest size before switching from a direct solver to GMRES.
  int max_full_;
  /// The rank enrichment size.
  int kickrank_;
  /// ALS enrichment size.
  int kick2_;
  /// Number of GMRES iterations for each subproblem.
  int local_iterations_;
  /// The number of restarts in GMRES.
  int resets_;
  /// Show output.
  bool verbose_;
  /// What preconditioner to use.
  int preconditioner_;

  // =================================================================
  // Protected constructors
  AMEnSolver(int nswp = 22, double eps = 1e-10,
    int max_rank = std::numeric_limits<int>::max(), int max_full = 500,
    int kickrank = 4, int kick2 = 0, int local_iterations = 40, int resets = 2,
    bool verbose = false, int preconditioner = 0)
    : nswp_(nswp), eps_(eps), max_rank_(max_rank), max_full_(max_full),
      kickrank_(kickrank), kick2_(kick2), local_iterations_(local_iterations),
      resets_(resets), verbose_(verbose)
  {
    if (nswp_ < 1 || eps_ < 0 || max_rank_ < 1 || max_full < 0 ||
        kickrank < 0 || kick2 < 0 || local_iterations_ < 1 || resets_ < 1) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`nswp`, `max_rank`, `local_iterations`, and `resets` must be greater\n"
        "than or equal to 1 and `eps`, `max_full`, `kickrank`, and `kick2`\n"
        "must be greater than or equal to 0");
    }

    if (preconditioner < 0 || preconditioner > 2) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`prec` is either 0, 1, or 2");
    }
  }

public:
  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new AMEn solver instance.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new AMEnSolver(std::forward<Args>(args)...));
  }

  /// @brief Solve the local linear system.
  /// @param local_system The local linear system to be solved.
  void solve(const linalg::LinearSystem::Ptr& local_system) override final;
};

} // namespace ttnte::solvers
