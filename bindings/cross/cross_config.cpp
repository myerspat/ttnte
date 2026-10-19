#include "ttnte/cross/cross_config.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_CrossConfig(py::module_& m)
{
  using namespace ttnte::cross;

  py::class_<CrossConfig>(m, "CrossConfig")
    .def(py::init(
           [](const ttnte::tensor::Layout& layout, double tol, bool weighted,
             std::vector<double> lethargy_widths, bool keep_tt, int64_t holdout,
             double holdout_factor, int64_t max_sweeps, int64_t max_rank,
             uint64_t seed, bool relax, double max_tol, int64_t num_threads,
             bool gram_svd, double inner_factor) {
             CrossConfig c;
             c.layout = layout;
             c.tol = tol;
             c.weighted = weighted;
             c.lethargy_widths = std::move(lethargy_widths);
             c.keep_tt = keep_tt;
             c.holdout = holdout;
             c.holdout_factor = holdout_factor;
             c.max_sweeps = max_sweeps;
             c.max_rank = max_rank;
             c.seed = seed;
             c.relax = relax;
             c.max_tol = max_tol;
             c.num_threads = num_threads;
             c.gram_svd = gram_svd;
             c.inner_factor = inner_factor;
             return c;
           }),
      py::arg("layout") = ttnte::tensor::Layout(), py::arg("tol") = 1e-4,
      py::arg("weighted") = true,
      py::arg("lethargy_widths") = std::vector<double> {},
      py::arg("keep_tt") = false, py::arg("holdout") = 16,
      py::arg("holdout_factor") = 10.0, py::arg("max_sweeps") = 8,
      py::arg("max_rank") = 0, py::arg("seed") = 0, py::arg("relax") = true,
      py::arg("max_tol") = 1e-2, py::arg("num_threads") = 16,
      py::arg("gram_svd") = false, py::arg("inner_factor") = 0.1)
    .def_readwrite("layout", &CrossConfig::layout,
      "Order and grouping of (polar, azimuthal, space, energy) into cores; a "
      "string or ttnte.tensor.Layout. Empty: polar|azimuthal|space|energy.")
    .def_readwrite("tol", &CrossConfig::tol)
    .def_readwrite("weighted", &CrossConfig::weighted)
    .def_readwrite("lethargy_widths", &CrossConfig::lethargy_widths)
    .def_readwrite("keep_tt", &CrossConfig::keep_tt)
    .def_readwrite("holdout", &CrossConfig::holdout)
    .def_readwrite("holdout_factor", &CrossConfig::holdout_factor)
    .def_readwrite("max_sweeps", &CrossConfig::max_sweeps)
    .def_readwrite("max_rank", &CrossConfig::max_rank)
    .def_readwrite("seed", &CrossConfig::seed)
    .def_readwrite("relax", &CrossConfig::relax)
    .def_readwrite("max_tol", &CrossConfig::max_tol)
    .def_readwrite("num_threads", &CrossConfig::num_threads,
      "Intra-op threads of the cross's linear algebra, capped by the solver's "
      "num_threads (<= 0: the solver's num_threads).")
    .def_readwrite("gram_svd", &CrossConfig::gram_svd,
      "Truncated SVDs of tall/wide superblocks through their Gram matrices "
      "(faster, singular values to about 1e-8 relative).")
    .def_readwrite("inner_factor", &CrossConfig::inner_factor,
      "Accuracy of each transport solve inside the sweep solver, as a "
      "fraction of the within-group-set tolerance (relaxed in GMRES).");
}
