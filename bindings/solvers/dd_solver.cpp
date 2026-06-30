#include "ttnte/solvers/dd_solver.hpp"
#include "../utils/label.hpp"
#include "ttnte/cad/patch.hpp"
#include "ttnte/mesh/mesh.hpp"
#include "ttnte/solvers/dd_strategy.hpp"
#include <torch/extension.h>

namespace py = pybind11;

template<typename BlockType>
static void register_DDSolver_impl(py::module_& m, const std::string& typestr)
{
  using DDSolver = ttnte::solvers::DDSolver<BlockType>;
  using SolverPtr = typename DDSolver::Ptr;

  std::string class_name = typestr + "DDSolver";

  register_Label<DDSolver>(m, class_name);

  py::class_<DDSolver, SolverPtr>(m, class_name.c_str())
    // =================================================================
    // Public constructors
    .def(py::init([](std::shared_ptr<ttnte::mesh::Mesh<BlockType>> mesh,
                    ttnte::solvers::DDStrategy::Ptr strategy,
                    std::optional<std::string> label) {
      return DDSolver::create(std::move(mesh), std::move(strategy), label);
    }),
      py::arg("mesh"), py::arg("strategy"), py::arg("label") = py::none())

    // =================================================================
    // Public methods
    .def("build_iteration_dag", &DDSolver::build_iteration_dag,
      "Build one iteration of the task graph using the stored strategy.",
      py::call_guard<py::gil_scoped_release>())
    .def("init", &DDSolver::init, py::arg("local_systems"),
      py::call_guard<py::gil_scoped_release>())
    .def("step", &DDSolver::step, py::call_guard<py::gil_scoped_release>())
    .def(
      "finalize", &DDSolver::finalize, py::call_guard<py::gil_scoped_release>())
    .def("is_initialized", &DDSolver::is_initialized)
    .def("is_finalized", &DDSolver::is_finalized)

    // =================================================================
    // Public getters / setters
    .def_property_readonly("label", &DDSolver::get_label)
    .def_property_readonly("strategy", &DDSolver::get_strategy)
    .def_property("local_systems", &DDSolver::get_local_systems,
      &DDSolver::set_local_systems)
    .def_property_readonly("gid_map", &DDSolver::get_gid_map)
    .def_property_readonly("world_comm", &DDSolver::get_world_comm)
    .def_property_readonly("boundary_comms", &DDSolver::get_boundary_comms)
    .def_property_readonly("stream_pool", &DDSolver::get_stream_pool);
}

void register_DDSolver(py::module_& m)
{
  register_DDSolver_impl<ttnte::cad::Patch>(m, "IGA");
}
