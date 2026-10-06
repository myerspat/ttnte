#include "ttnte/linalg/linear_system.hpp"
#include "../utils/label.hpp"
#include <limits>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_LinearSystem(py::module_& m)
{
  using namespace ttnte::linalg;

  register_Label<LinearSystem>(m, "LinearSystem");
  py::class_<LinearSystem, std::shared_ptr<LinearSystem>> ls(
    m, "LinearSystem", "ttnte abstract linear system");

  // =================================================================
  // Constructors (Factory pattern using lambda)
  ls.def(py::init([](Operator interior_op,
                    std::vector<NeighborCoupling> couplings_vec, State state,
                    Source::Ptr source, std::optional<std::string> label,
                    Operator moment_projector, Operator scatter_op) {
    c10::SmallVector<NeighborCoupling, 6> couplings(
      couplings_vec.begin(), couplings_vec.end());
    return LinearSystem::create(std::move(interior_op), std::move(couplings),
      std::move(state), std::move(source), std::move(label),
      std::move(moment_projector), std::move(scatter_op));
  }),
    py::arg("interior_op"),
    py::arg("couplings") = std::vector<NeighborCoupling> {},
    py::arg("state") = State(), py::arg("source") = nullptr,
    py::arg("label") = py::none(), py::arg("moment_projector") = Operator(),
    py::arg("scatter_op") = Operator(),
    "Constructs a LinearSystem via the create factory method");

  // =================================================================
  // Properties & Getters/Setters (Keep GIL: Trivial accesses)
  ls.def_property_readonly(
      "label", [](const LinearSystem& l) { return l.get_label().to_string(); })
    .def_property_readonly("interior_op", &LinearSystem::get_interior_op)
    .def_property_readonly(
      "moment_projector", &LinearSystem::get_moment_projector)
    .def_property_readonly("scatter_op", &LinearSystem::get_scatter_op)
    .def_property("state", &LinearSystem::get_state, &LinearSystem::set_state)
    .def_property_readonly("source", &LinearSystem::get_source)
    .def_property_readonly("dtype", &LinearSystem::get_dtype)
    .def_property("gid", &LinearSystem::get_gid, &LinearSystem::set_gid)
    .def_property_readonly(
      "couplings",
      [](LinearSystem& ls) {
        const auto& couplings = ls.get_couplings();
        return std::vector<NeighborCoupling>(
          couplings.begin(), couplings.end());
      },
      "Get the couplings as a Python list. Returns a COPY (c10::SmallVector "
      "has no pybind11 STL caster, unlike std::vector -- same pattern as "
      "DGFirstOrderTransportAssembler::get_outflow_ops/get_inflow_ops/"
      "current_ops); mutating entries in the returned list does not affect "
      "this LinearSystem's own couplings.")
    .def(
      "set_coupling_recv_buffer",
      [](LinearSystem& ls, size_t idx, State face) {
        auto& couplings = ls.get_couplings();
        if (idx >= couplings.size()) {
          throw std::out_of_range("coupling index out of range");
        }
        couplings[idx].recv_buffer = std::move(face);
      },
      py::arg("idx"), py::arg("face"),
      "Set the pending (raw, unmapped) neighbor face for coupling idx on the "
      "live coupling (the `couplings` property returns copies). The next "
      "step's apply task maps it and applies boundary_op.")
    .def("add_coupling", &LinearSystem::add_coupling, py::arg("coupling"),
      "Append a coupling after construction. The coupling's boundary_op will "
      "not be in the flat buffer; call before any transfer_buffer.");

  ls.def(
    "update_source",
    [](LinearSystem& ls, const State& eigvec, double eps, int64_t max_rank) {
      ls.update_source(eigvec, eps, max_rank);
    },
    py::arg("eigvec"), py::arg("eps") = 1e-12,
    py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
    py::call_guard<py::gil_scoped_release>(),
    "Recompute the source state from the current solution iterate (no-op for "
    "fixed sources).");

  ls.def("get_buffer", &LinearSystem::get_buffer, py::arg("device"),
    py::return_value_policy::reference_internal,
    "Returns the underlying buffer tensor for a given device.");

  // =================================================================
  // Device & Dtype Transfers (Release GIL: High-latency memory IO)

  // transfer_buffer overloads
  ls.def("transfer_buffer",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(
        &LinearSystem::transfer_buffer),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("transfer_buffer",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_buffer),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("transfer_buffer",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_buffer),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());

  // transfer_nonbuffer overloads
  ls.def("transfer_nonbuffer",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(
        &LinearSystem::transfer_nonbuffer),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("transfer_nonbuffer",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_nonbuffer),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("transfer_nonbuffer",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::transfer_nonbuffer),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());

  // to_ overloads (In-place linear system cast)
  ls.def("to_",
      py::overload_cast<const torch::TensorOptions&, bool, bool>(
        &LinearSystem::to_),
      py::arg("options"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&LinearSystem::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none(),
      py::call_guard<py::gil_scoped_release>());
}
