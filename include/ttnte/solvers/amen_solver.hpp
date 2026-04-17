#pragma once

#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::solvers {

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
  double eps_;
  int rmax_;
  int max_full_;
  int kickrank_;
  int kick2_;
  int local_iterations_;
  int resets_;
  bool verbose_;
  int prec_;

  // =================================================================
  // Protected constructors
  AMEnSolver(int nswp = 22, double eps = 1e-10,
    int rmax = std::numeric_limits<int>::max(), int max_full = 500,
    int kickrank = 4, int kick2 = 0, int local_iterations = 40, int resets = 2,
    bool verbose = false, int prec = 0)
    : nswp_(nswp), eps_(eps), rmax_(rmax), max_full_(max_full),
      kickrank_(kickrank), kick2_(kick2), local_iterations_(local_iterations),
      resets_(resets), verbose_(verbose)
  {
    if (nswp_ < 1 || eps_ < 0 || rmax_ < 1 || max_full < 0 || kickrank < 0 ||
        kick2 < 0 || local_iterations_ < 1 || resets_ < 1) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`nswp`, `rmax`, `local_iterations`, and `resets` must be greater\n"
        "than or equal to 1 and `eps`, `max_full`, `kickrank`, and `kick2`\n"
        "must be greater than or equal to 0");
    }

    if (prec < 0 || prec > 2) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`prec` is either 0, 1, or 2");
    }
  }

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new AMEnSolver(std::forward<Args>(args)...));
  }

  void solve(const linalg::LinearSystem::Ptr& local_system) override final;
};

} // namespace ttnte::solvers
