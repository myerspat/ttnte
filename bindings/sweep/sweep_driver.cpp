#include "ttnte/sweep/sweep_driver.hpp"
#include "ttnte/cad/patch.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

template<typename BlockType, int64_t NumDim>
static void register_SweepDriver_impl(
  py::module_& m, const std::string& typestr)
{
  using SweepDriver = ttnte::sweep::SweepDriver<BlockType, NumDim>;
  std::string class_name =
    typestr + "SweepDriver" + std::to_string(NumDim) + "D";

  py::class_<SweepDriver, typename SweepDriver::Ptr>(m, class_name.c_str())
    // =================================================================
    // Public constructors
    .def(py::init([](typename SweepDriver::Mesh::Ptr mesh,
                    ttnte::xs::Server::Ptr xs_server,
                    const ttnte::parallel::ParallelContext& mpi_context) {
      return SweepDriver::create(
        std::move(mesh), std::move(xs_server), mpi_context);
    }),
      py::arg("mesh"), py::arg("xs_server"), py::arg("mpi_context"))

    // =================================================================
    // Public methods
    .def("setup", &SweepDriver::setup,
      "Build the patch discretizations, sweep graph and task DAG.",
      py::arg("angular_qset"), py::arg("config"),
      py::call_guard<py::gil_scoped_release>())
    .def("sweep", &SweepDriver::sweep,
      "Run one transport sweep with the current moments as the scattering "
      "source.",
      py::call_guard<py::gil_scoped_release>())
    .def("solve_fixed_source", &SweepDriver::solve_fixed_source,
      "Solve a fixed-source problem with unaccelerated source iteration. "
      "Returns a TransportSolution holding the angular flux "
      "(store_angular_flux=True) or the scalar flux.",
      py::call_guard<py::gil_scoped_release>())
    .def("build_solution", &SweepDriver::build_solution)
    .def("get_scalar_flux", &SweepDriver::get_scalar_flux, py::arg("gid"),
      "Scalar flux coefficients (*ctrlpts_grid, G).")
    .def("get_moments", &SweepDriver::get_moments, py::arg("gid"),
      "Flux moments (num_moments, G, N).")
    .def("get_angular_flux", &SweepDriver::get_angular_flux, py::arg("gid"),
      "Angular flux (*angular_modes, *ctrlpts_grid, G) in TT core order.")
    .def("get_discretization", &SweepDriver::get_discretization, py::arg("gid"))
    .def("assemble_dense_operator", &SweepDriver::assemble_dense_operator,
      py::arg("gid"),
      "Dense patch operator (reflective implicit, interfaces excluded) in TT "
      "ordering (direction, dofs, group). Verification only.")

    // =================================================================
    // Public getters
    .def_property_readonly("config", &SweepDriver::get_config)
    .def_property_readonly("graph", &SweepDriver::get_graph)
    .def_property_readonly("mesh", &SweepDriver::get_mesh)
    .def_property_readonly("harmonics", &SweepDriver::get_harmonics)
    .def_property_readonly("ordinates", &SweepDriver::get_ordinates)
    .def_property_readonly("weights", &SweepDriver::get_weights)
    .def_property_readonly("angular_modes", &SweepDriver::angular_modes)
    .def_property_readonly("errors", &SweepDriver::get_errors)
    .def_property_readonly("sweep_times", &SweepDriver::get_sweep_times)
    .def_property_readonly("num_iterations", &SweepDriver::get_num_iterations)
    .def_property_readonly("converged", &SweepDriver::is_converged)
    .def_property_readonly("num_tasks", &SweepDriver::num_tasks);
}

void register_SweepDriver(py::module_& m)
{
  register_SweepDriver_impl<ttnte::cad::Patch, 1>(m, "IGA");
  register_SweepDriver_impl<ttnte::cad::Patch, 2>(m, "IGA");
  register_SweepDriver_impl<ttnte::cad::Patch, 3>(m, "IGA");
}
