#include "ttnte/sweep/sweep_driver.hpp"
#include "ttnte/cad/patch.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include <pybind11/functional.h>
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
    .def(py::init(
           [](typename SweepDriver::Mesh::Ptr mesh,
             ttnte::xs::Server::Ptr xs_server,
             const ttnte::parallel::ParallelContext& mpi_context, bool serial) {
             return SweepDriver::create(
               std::move(mesh), std::move(xs_server), mpi_context, serial);
           }),
      py::arg("mesh"), py::arg("xs_server"), py::arg("mpi_context"),
      py::arg("serial") = false,
      "serial=True solves the whole mesh on this rank alone (MPI_COMM_SELF); "
      "otherwise setup() partitions the mesh over all ranks.")

    // =================================================================
    // Public methods
    .def("setup", &SweepDriver::setup,
      "Build the patch discretizations, sweep graph and task DAG. `chunks` "
      "optionally maps GID -> chunk id to group units into tasks (e.g. FDG "
      "elements by parent patch); empty uses config.task_granularity.",
      py::arg("angular_qset"), py::arg("config"),
      py::arg("chunks") = std::unordered_map<int64_t, int64_t> {},
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
    .def("solve_eigenvalue", &SweepDriver::solve_eigenvalue,
      "Solve the k-eigenvalue problem by power iteration (as "
      "TransportDriver.solve_eigenvalue). Returns a TransportSolution "
      "holding k_eff.",
      py::arg("k_tol") = 1e-6, py::call_guard<py::gil_scoped_release>())
    .def("build_solution", &SweepDriver::build_solution)
    .def("set_partition", &SweepDriver::set_partition, py::arg("gid_to_rank"),
      "Use a fixed partition {GID: MPI rank} for multi-rank sweeps instead "
      "of METIS (takes effect at the next setup()).")
    .def(
      "set_callback",
      [](SweepDriver& self,
        std::optional<typename SweepDriver::Callback> callback,
        int64_t frequency) {
        if (!callback.has_value()) {
          self.set_callback(typename SweepDriver::Callback(), frequency);
          return;
        }
        self.set_callback(
          [callback = std::move(*callback)](const SweepDriver& driver) {
            py::gil_scoped_acquire acquire;
            callback(driver);
          },
          frequency);
      },
      py::arg("callback"), py::arg("frequency") = 1,
      "Set a callback invoked with this driver after every `frequency`-th "
      "outer iteration of solve_fixed_source()/solve_eigenvalue(). Callback "
      "time is excluded from solve_seconds. Pass None to disable.")
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
    .def_property_readonly("k_eff", &SweepDriver::get_k_eff)
    .def_property_readonly("solve_seconds", &SweepDriver::get_solve_seconds,
      "Wall seconds of the solve up to its last completed outer iteration, "
      "excluding callback time.")
    .def_property_readonly("k_history", &SweepDriver::get_k_history)
    .def_property_readonly("outer_sweeps", &SweepDriver::get_outer_sweeps,
      "Group-set sweeps of each outer iteration of the last solve.")
    .def_property_readonly(
      "group_set_sweeps", &SweepDriver::get_group_set_sweeps)
    .def_property_readonly("equivalent_sweeps",
      &SweepDriver::get_equivalent_sweeps,
      "Full-sweep equivalents of the last solve (sum of |set| / G).")
    .def_property_readonly(
      "num_delayed_blocks", &SweepDriver::num_delayed_blocks)
    .def_property_readonly("converged", &SweepDriver::is_converged)
    .def_property_readonly("num_tasks", &SweepDriver::num_tasks)
    .def_property_readonly("chunks", &SweepDriver::get_chunks)
    .def_property_readonly("factorization_bytes",
      &SweepDriver::factorization_bytes,
      "Bytes held by the factorization cache (0 if disabled).")
    .def_property_readonly("dag_seconds", &SweepDriver::get_dag_seconds,
      "Wall seconds spent executing sweep DAGs (parallel part)")
    .def_property_readonly("task_seconds", &SweepDriver::get_task_seconds,
      "Cumulative seconds inside sweep task bodies, summed over threads.")
    .def("reset_task_seconds", &SweepDriver::reset_task_seconds);
}

void register_SweepDriver(py::module_& m)
{
  register_SweepDriver_impl<ttnte::cad::Patch, 1>(m, "IGA");
  register_SweepDriver_impl<ttnte::cad::Patch, 2>(m, "IGA");
  register_SweepDriver_impl<ttnte::cad::Patch, 3>(m, "IGA");
}
