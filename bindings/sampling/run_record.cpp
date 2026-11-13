#include "ttnte/sampling/run_record.hpp"
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_RunRecord(py::module_& m)
{
  using namespace ttnte::sampling;

  py::class_<RunRecord, RunRecord::Ptr>(m, "RunRecord",
    "Record of one sampled transport solve, common to every sampler; a "
    "sampler's records are subclasses (CrossRun, GalerkinRun) adding its "
    "own.")
    .def_readonly("tolerance", &RunRecord::tolerance,
      "Relative tolerance the run was asked for")
    .def_readonly("fibers", &RunRecord::fibers, "Fibers it swept")
    .def_readonly("ranks", &RunRecord::ranks, "Ranks of its approximation")
    .def_property_readonly(
      "max_rank",
      [](const RunRecord& r) {
        int64_t out = 0;
        for (int64_t v : r.ranks)
          out = std::max(out, v);
        return out;
      },
      "Largest of the ranks")
    .def_readonly("error", &RunRecord::error,
      "Its stopping test's error estimate (judged against the tolerance)")
    .def_readonly("converged", &RunRecord::converged)
    .def_readonly("seconds", &RunRecord::seconds)
    .def_readonly("sweep_seconds", &RunRecord::sweep_seconds);
}
