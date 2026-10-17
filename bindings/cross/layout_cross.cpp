#include "ttnte/cross/layout_cross.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_LayoutCross(py::module_& m)
{
  using namespace ttnte::cross;

  py::class_<LayoutCross, LayoutCross::Ptr>(m, "LayoutCross",
    "Sweep-sampled TT-cross of one transport solve. For standalone use the "
    "oracle `sweep(blocks)` is a Python callable taking a list of "
    "(directions, groups) and returning a (num_fibers, num_space) tensor, "
    "block by block, direction-major then group.")
    .def(py::init([](int64_t num_dim, int64_t num_polar, int64_t num_azimuthal,
                    int64_t num_groups, int64_t num_space,
                    torch::Tensor weights, torch::Tensor harmonics,
                    std::vector<int64_t> initial_directions, py::function sweep,
                    const CrossConfig& config,
                    std::vector<int64_t> patch_offsets,
                    std::vector<std::vector<int64_t>> patch_shapes,
                    std::vector<torch::Tensor> mass_bands) {
      CrossProblem p;
      p.num_dim = num_dim;
      p.num_polar = num_polar;
      p.num_azimuthal = num_azimuthal;
      p.num_groups = num_groups;
      p.num_space = num_space;
      p.weights = weights;
      p.harmonics = harmonics;
      p.initial_directions = std::move(initial_directions);
      p.patch_offsets =
        patch_offsets.empty() ? std::vector<int64_t> {0} : patch_offsets;
      p.patch_shapes = patch_shapes.empty()
                         ? std::vector<std::vector<int64_t>> {{num_space}}
                         : patch_shapes;
      p.mass_bands = std::move(mass_bands);
      p.sweep = [sweep, num_space](
                  const std::vector<ttnte::sweep::FiberBlock>& blocks,
                  double* out) {
        py::gil_scoped_acquire gil;
        py::list request;
        int64_t count = 0;
        for (const auto& b : blocks) {
          request.append(py::make_tuple(b.directions, b.groups));
          count += static_cast<int64_t>(b.directions.size() * b.groups.size());
        }
        auto result = sweep(request).cast<torch::Tensor>();
        result = result.to(torch::kFloat64).contiguous();
        if (result.numel() != count * num_space)
          throw std::runtime_error("sweep returned the wrong number of values");
        std::copy(result.data_ptr<double>(),
          result.data_ptr<double>() + result.numel(), out);
      };
      return LayoutCross::create(std::move(p), config);
    }),
      py::arg("num_dim"), py::arg("num_polar"), py::arg("num_azimuthal"),
      py::arg("num_groups"), py::arg("num_space"), py::arg("weights"),
      py::arg("harmonics"), py::arg("initial_directions"), py::arg("sweep"),
      py::arg("config") = CrossConfig(),
      py::arg("patch_offsets") = std::vector<int64_t> {},
      py::arg("patch_shapes") = std::vector<std::vector<int64_t>> {},
      py::arg("mass_bands") = std::vector<torch::Tensor> {})
    .def("run", &LayoutCross::run)
    .def("moments", &LayoutCross::moments,
      "(num_moments, G, num_space) moments of the last run")
    .def("evaluate", &LayoutCross::evaluate, py::arg("directions"),
      py::arg("groups"), py::arg("space"),
      "(|directions|, |groups|, |space|) values of the angular flux TT")
    .def("patch_tt", &LayoutCross::patch_tt, py::arg("patch"),
      "Cores of one patch's TT in the configured layout")
    .def_property_readonly("cores", &LayoutCross::cores,
      "Global TT cores (r_l, n_k, r_r) in the solver layout (space merged)")
    .def_property_readonly(
      "layout", [](const LayoutCross& c) { return c.layout().to_string(); })
    .def_property_readonly("user_layout",
      [](const LayoutCross& c) { return c.user_layout().to_string(); })
    .def_property_readonly("ranks", &LayoutCross::ranks)
    .def_property_readonly(
      "history",
      [](const LayoutCross& c) {
        py::list out;
        for (const auto& h : c.history())
          out.append(
            py::dict(py::arg("phase") = h.forward ? "forward" : "backward",
              py::arg("ranks") = h.ranks, py::arg("fibers") = h.fibers,
              py::arg("holdout_error") = h.holdout_error));
        return out;
      },
      "Per half-sweep of the last run: phase, ranks, fibers swept so far, "
      "hold-out error (backward only)")
    .def_property_readonly("num_fibers", &LayoutCross::num_fibers)
    .def_property_readonly("total_fibers", &LayoutCross::total_fibers)
    .def_property_readonly("holdout_error", &LayoutCross::holdout_error)
    .def_property_readonly("converged", &LayoutCross::converged)
    .def_property_readonly("sweeps_done", &LayoutCross::sweeps_done)
    .def_property_readonly("sweep_seconds", &LayoutCross::sweep_seconds)
    .def_property_readonly("seconds", &LayoutCross::seconds);
}
