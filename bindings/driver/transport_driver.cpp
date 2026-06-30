#include "ttnte/driver/transport_driver.hpp"
#include "../utils/label.hpp"
#include "ttnte/cad/patch.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

template<typename BlockType, int64_t NumDim>
static void register_TransportDriver_impl(
  py::module_& m, const std::string& typestr)
{
  using TransportDriver = ttnte::driver::TransportDriver<BlockType, NumDim>;
  using LoadHeuristicPtr = typename TransportDriver::LoadHeuristicPtr;
  using DriverPtr = typename TransportDriver::Ptr;

  std::string class_name =
    typestr + "TransportDriver" + std::to_string(NumDim) + "D";
  register_Label<TransportDriver>(m, class_name);

  py::class_<TransportDriver, DriverPtr>(m, class_name.c_str())
    // =================================================================
    // Public constructors
    .def(py::init([](typename TransportDriver::Mesh::Ptr mesh,
                    ttnte::xs::Server::Ptr xs_server,
                    const ttnte::parallel::ParallelContext& mpi_context,
                    std::optional<std::string> label) {
      return TransportDriver::create(
        std::move(mesh), std::move(xs_server), mpi_context, label);
    }),
      py::arg("mesh"), py::arg("xs_server"), py::arg("mpi_context"),
      py::arg("label") = std::nullopt)

    // =================================================================
    // Public methods
    .def("assemble", &TransportDriver::assemble,
      "Assemble the linear system for each local mesh block.",
      py::arg("angular_qset"), py::arg("config"),
      py::call_guard<py::gil_scoped_release>())
    .def("init_solver", &TransportDriver::init_solver,
      "Set up the DDSolver, build the iteration DAG, and compute the initial "
      "fission source. Returns the initial k-eigenvalue.",
      py::arg("strategy"), py::call_guard<py::gil_scoped_release>())
    .def("solve_eigenvalue", &TransportDriver::solve_eigenvalue,
      "Run the k-eigenvalue power iteration with the given DD strategy. "
      "Returns the converged k-effective.",
      py::arg("strategy"), py::arg("tol") = 1e-8, py::arg("max_iter") = 500,
      py::arg("verbose") = true, py::call_guard<py::gil_scoped_release>())
    .def("distribute", &TransportDriver::distribute,
      "Initial partition using METIS on rank 0 and cull the local mesh.",
      py::arg("load_heuristics") = std::vector<LoadHeuristicPtr> {},
      py::arg("root_rank") = 0, py::call_guard<py::gil_scoped_release>())
    .def("redistribute", &TransportDriver::redistribute,
      "Dynamic repartitioning using ParMETIS.",
      py::arg("load_heuristics") = std::vector<LoadHeuristicPtr> {},
      py::call_guard<py::gil_scoped_release>())
    .def("get_assembler", &TransportDriver::get_assembler,
      "Return the assembler for a mesh block GID (throws if cleared).",
      py::arg("gid"))
    .def("get_system", &TransportDriver::get_system,
      "Return the linear system for a mesh block GID.", py::arg("gid"))

    // =================================================================
    // Public getters / setters
    .def_property(
      "label", &TransportDriver::get_label, &TransportDriver::set_label)
    .def_property_readonly("mesh", &TransportDriver::get_mesh)
    .def_property_readonly("server", &TransportDriver::get_server)
    .def_property_readonly("solver", &TransportDriver::get_solver);
}

void register_TransportDriver(py::module_& m)
{
  register_TransportDriver_impl<ttnte::cad::Patch, 1>(m, "IGA");
  register_TransportDriver_impl<ttnte::cad::Patch, 2>(m, "IGA");
  register_TransportDriver_impl<ttnte::cad::Patch, 3>(m, "IGA");
}
