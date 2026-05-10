#pragma once

#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/python/acquire.hpp"
#include <limits>
#include <optional>

namespace ttnte::python::torchtt {

class Acquire : public python::Acquire {
public:
  Acquire();
  ~Acquire() override;

  /// @brief Compute a TT element-wise division using torchTT's implementation
  /// of the AMEn divide algorithm. Calls `torchtt._division.amen_divide()`.
  linalg::TTEngine amen_divide(const linalg::TTEngine& a,
    const linalg::TTEngine& b, int nswp = 22,
    std::optional<linalg::TTEngine> x0 = std::nullopt, double eps = 1e-10,
    int rmax = std::numeric_limits<int>::max(), int max_full = 500,
    int kickrank = 4, int kick2 = 0, std::string trunc_norm = "res",
    int local_iterations = 40, int resets = 2, bool verbose = true,
    std::optional<std::string> preconditioner = std::nullopt);

  /// @brief Compute a TT-matrix by TT-matrix product using AMEn. This method
  /// calls `torchtt._amen.amen_mv`.
  linalg::TTEngine amen_mm(const linalg::TTEngine& a, const linalg::TTEngine& b,
    int nswp = 22, std::optional<linalg::TTEngine> x0 = std::nullopt,
    double eps = 1e-10, int rmax = std::numeric_limits<int>::max(),
    int kickrank = 4, int kick2 = 0, bool verbose = false);
};

} // namespace ttnte::python::torchtt
