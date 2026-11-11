#include "ttnte/cross/cross_config.hpp"
#include "../utils/kwargs_init.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_CrossConfig(py::module_& m)
{
  using namespace ttnte::cross;

  py::class_<CrossConfig>(m, "CrossConfig")
    // Keyword arguments only, each an attribute below; defaults from C++
    .def(ttnte::bindings::kwargs_init<CrossConfig>())
    .def_readwrite("layout", &CrossConfig::layout,
      "Order and grouping of (polar, azimuthal, space, energy) into cores; a "
      "string or ttnte.tensor.Layout. Empty: polar|azimuthal|space|energy.")
    .def_readwrite("tol", &CrossConfig::tol)
    .def_readwrite("weighted", &CrossConfig::weighted)
    .def_readwrite("keep_angular_flux", &CrossConfig::keep_angular_flux)
    .def_readwrite("holdout", &CrossConfig::holdout)
    .def_readwrite("resampled_pool", &CrossConfig::resampled_pool,
      "Pool of swept fibers judged once (nearly) every pair is swept, in "
      "units of holdout.")
    .def_readwrite("swept_sample", &CrossConfig::swept_sample,
      "Swept fibers sampled for the global estimate's in-sample stratum, in "
      "units of holdout.")
    .def_readwrite("forced_per_check", &CrossConfig::forced_per_check,
      "Worst pool pairs forced into the index sets after a failed check.")
    .def_readwrite("holdout_factor", &CrossConfig::holdout_factor)
    .def_readwrite("global_holdout", &CrossConfig::global_holdout,
      "Accept on the estimated global relative error (default) instead of "
      "the pool's error relative to its own norm (both recorded).")
    .def_readwrite("max_sweeps", &CrossConfig::max_sweeps)
    .def_readwrite("stall_passes", &CrossConfig::stall_passes,
      "Stop a run, unconverged, after this many consecutive passes with "
      "unchanged ranks and a hold-out error lowered by less than "
      "stall_improvement (0: never).")
    .def_readwrite("stall_improvement", &CrossConfig::stall_improvement)
    .def_readwrite("max_rank", &CrossConfig::max_rank)
    .def_readwrite("seed", &CrossConfig::seed)
    .def_readwrite("gram_floor", &CrossConfig::gram_floor,
      "Smallest relative truncation threshold taken through the Gram matrix.")
    .def_readwrite("forward_truncation_share",
      &CrossConfig::forward_truncation_share,
      "Forward-pass truncation as a fraction of the per-bond share of tol.")
    .def_readwrite("gram_solve_min_ratio", &CrossConfig::gram_solve_min_ratio,
      "Smallest sigma_min / sigma_max inverted through the Gram matrix; "
      "below it the pseudo-inverse.")
    .def_readwrite("growth_share", &CrossConfig::growth_share,
      "Share of the last core's residual energy that bond growth covers.")
    .def_readwrite("growth_max_blocks", &CrossConfig::growth_max_blocks,
      "Most blocks added by one bond growth, or half the set's blocks if "
      "larger.")
    .def_readwrite("sketch_rows", &CrossConfig::sketch_rows,
      "Rows of the first sketch choosing the initial energies.")
    .def_readwrite("gram_svd", &CrossConfig::gram_svd,
      "Truncated SVDs of tall/wide superblocks through their Gram matrices "
      "(faster, singular values to about 1e-8 relative).")
    .def_readwrite("warm_start", &CrossConfig::warm_start,
      "Start each run from the previous run's right index sets.")
    .def_readwrite("select_cond", &CrossConfig::select_cond,
      "Index selection stops once the chosen rows have sigma_min / sigma_max "
      ">= select_cond, in (0, 1).")
    .def_readwrite("select_near", &CrossConfig::select_near,
      "Among blocks scoring at least select_near times the best, the "
      "selection takes the cheapest, in (0, 1].");
}
