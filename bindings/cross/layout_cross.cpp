#include "ttnte/cross/layout_cross.hpp"
#include "ttnte/utils/exception.hpp"
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
                    std::vector<torch::Tensor> mass_bands,
                    std::vector<double> lethargy_widths) {
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
      p.lethargy_widths = std::move(lethargy_widths);
      p.sweep = [sweep, num_space](
                  const std::vector<ttnte::sweep::FiberBlock>& blocks,
                  double* out, const ttnte::sweep::FiberDone& done) {
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
          throw ttnte::utils::runtime_error(
            "ttnte::cross::LayoutCross (Python oracle)",
            "sweep returned the wrong number of values");
        std::copy(result.data_ptr<double>(),
          result.data_ptr<double>() + result.numel(), out);
        if (done)
          done(0, count);
      };
      return LayoutCross::create(std::move(p), config);
    }),
      py::arg("num_dim"), py::arg("num_polar"), py::arg("num_azimuthal"),
      py::arg("num_groups"), py::arg("num_space"), py::arg("weights"),
      py::arg("harmonics"), py::arg("initial_directions"), py::arg("sweep"),
      py::arg("config") = CrossConfig(),
      py::arg("patch_offsets") = std::vector<int64_t> {},
      py::arg("patch_shapes") = std::vector<std::vector<int64_t>> {},
      py::arg("mass_bands") = std::vector<torch::Tensor> {},
      py::arg("lethargy_widths") = std::vector<double> {})
    .def("run", &LayoutCross::run)
    .def("moments", &LayoutCross::moments,
      "(num_moments, G, num_space) moments of the last run")
    .def("evaluate_pairs", &LayoutCross::evaluate_pairs, py::arg("pairs"),
      "(|pairs|, num_space) fibers of the TT at pairs d * G + g")
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
    // Testing aids
    .def("index_set", &LayoutCross::index_set, py::arg("left"), py::arg("k"),
      "(|set|, 4) components (mu, gamma, s, E; -1 free) of left_[k] or "
      "right_[k] after the last run (testing aid)")
    .def("core_components", &LayoutCross::core_components, py::arg("k"),
      "(n_k, 4) components of every value of core k (testing aid)")
    .def("block_key", &LayoutCross::block_key, py::arg("mu"), py::arg("ga"),
      py::arg("e"), "Block key of (mu, gamma, E), -1 free (testing aid)")
    .def("unswept_in_block", &LayoutCross::unswept_in_block, py::arg("key"),
      "Fibers of a block not in the store (testing aid)")
    .def_property_readonly("forced_pairs", &LayoutCross::forced_pairs,
      "Pairs forced into the index sets in the last run (testing aid)")
    .def("forced_block_keys", &LayoutCross::forced_block_keys, py::arg("first"),
      py::arg("last"),
      "Forced pairs projected to the modes of cores first..last (testing aid)")
    .def("grown_block_keys", &LayoutCross::grown_block_keys, py::arg("first"),
      py::arg("last"),
      "Block keys the residual growth added for cores first..last in the last "
      "run (testing aid)")
    .def("candidate_grams", &LayoutCross::candidate_grams, py::arg("k"),
      py::arg("left"), py::arg("core"), py::arg("factored") = false,
      "(block Grams (nb, r, r), block keys) of core k's candidates for a "
      "given core against the current side basis, by selection's path or "
      "all factored (testing aid)")
    .def("set_factored_grams", &LayoutCross::set_factored_grams, py::arg("on"),
      "Form every block Gram in factored form during selection (testing aid)")
    .def("set_recording", &LayoutCross::set_recording, py::arg("on"),
      "Record every half pass of the next runs (testing aid)")
    .def_property_readonly("num_recorded", &LayoutCross::num_recorded,
      "Half passes recorded in the last run (testing aid)")
    .def("recorded_forward", &LayoutCross::recorded_forward, py::arg("i"),
      "Whether recorded half pass i was forward (testing aid)")
    .def("recorded_cores", &LayoutCross::recorded_cores, py::arg("i"),
      "Cores after recorded half pass i, before unweighting (testing aid)")
    .def("recorded_bases", &LayoutCross::recorded_bases, py::arg("i"),
      "Bases after recorded half pass i: B_ forward, C_ backward, None where "
      "not chosen (testing aid)")
    .def("recorded_index_set", &LayoutCross::recorded_index_set, py::arg("i"),
      py::arg("left"), py::arg("k"),
      "index_set() after recorded half pass i (testing aid)")
    .def_property_readonly(
      "history",
      [](const LayoutCross& c) {
        py::list out;
        for (const auto& h : c.history())
          out.append(
            py::dict(py::arg("phase") = h.forward ? "forward" : "backward",
              py::arg("ranks") = h.ranks, py::arg("fibers") = h.fibers,
              py::arg("holdout_error") = h.holdout_error,
              py::arg("holdout_rms") = h.holdout_rms,
              py::arg("swept_rms") = h.swept_rms,
              py::arg("global_error") = h.global_error,
              py::arg("forced") = h.forced));
        return out;
      },
      "Per half-sweep of the last run: phase, ranks, fibers swept so far, "
      "and (backward only) hold-out error and the RMS norms of the hold-out "
      "pool and of all swept fibers (weighted)")
    .def_property_readonly("num_fibers", &LayoutCross::num_fibers)
    .def_property_readonly("total_fibers", &LayoutCross::total_fibers)
    .def_property_readonly("holdout_error", &LayoutCross::holdout_error,
      "Error the last check was accepted or rejected on: the pool's, or the "
      "global estimate when the pool failed and global_holdout is set")
    .def_property_readonly("global_error", &LayoutCross::global_error,
      "Estimated global relative error at the last check, NaN if the pool "
      "alone accepted it")
    .def_property_readonly("converged", &LayoutCross::converged)
    .def_property_readonly("stalled", &LayoutCross::stalled,
      "Whether the last run stopped on stalled passes (stall_passes)")
    .def_property_readonly("sweeps_done", &LayoutCross::sweeps_done)
    .def_property_readonly("sweep_seconds", &LayoutCross::sweep_seconds)
    .def_property_readonly("seconds", &LayoutCross::seconds)
    .def_property_readonly("timings", &LayoutCross::timings)
    .def_property_readonly(
      "sweep_calls",
      [](const LayoutCross& c) {
        py::list out;
        for (const auto& s : c.sweep_calls())
          out.append(py::dict(py::arg("phase") = s.phase,
            py::arg("fibers") = s.fibers, py::arg("blocks") = s.blocks,
            py::arg("directions") = s.directions,
            py::arg("seconds") = s.seconds));
        return out;
      },
      "Every sweep request of every run: phase (init, superblock, check), "
      "new fibers, sweep blocks, distinct directions, seconds");
}
