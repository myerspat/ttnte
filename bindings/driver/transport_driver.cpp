#include "ttnte/driver/transport_driver.hpp"
#include "../utils/label.hpp"
#include "ttnte/cad/patch.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include <pybind11/functional.h>
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
      "Set up the solver, build its iteration DAG, and compute the initial "
      "fission source. Returns the initial k-eigenvalue.",
      py::arg("solver"), py::arg("clear_assemblers") = true,
      py::call_guard<py::gil_scoped_release>())
    .def("solve_eigenvalue", &TransportDriver::solve_eigenvalue,
      "Run the k-eigenvalue power iteration with the given solver (e.g. a "
      "DDSolver for multi-patch domain decomposition, or a bare LocalSolver "
      "such as AMEnSolver for a single-patch problem). Returns a "
      "TransportSolution holding the converged k-effective (.k_eff) and the "
      "raw angular flux per local patch. Convergence requires BOTH the "
      "scalar-flux-shape relative L2 error (tol) and k_eff's own absolute "
      "iteration-to-iteration change (k_tol, in k-units -- e.g. 1e-5 is 1 "
      "pcm) to fall below their respective tolerances.",
      py::arg("inner_solver"), py::arg("tol") = 1e-8, py::arg("max_iter") = 500,
      py::arg("clear_assemblers") = true, py::arg("verbose") = true,
      py::arg("k_tol") = 1e-5, py::call_guard<py::gil_scoped_release>())
    .def("solve_fixed_source", &TransportDriver::solve_fixed_source,
      "Run the fixed-source solver with the given solver (e.g. a DDSolver "
      "for multi-patch domain decomposition, or a bare LocalSolver such as "
      "AMEnSolver for a single-patch problem). Unlike solve_eigenvalue(), "
      "there is no eigenvalue to update each outer iteration -- every "
      "attached Source is already fixed at assembly time. Returns a "
      "TransportSolution holding the raw angular flux per local patch "
      "(k_eff is unset). Convergence requires the scalar-flux-shape "
      "relative L2 error (tol) between successive outer iterations to fall "
      "below tol.",
      py::arg("inner_solver"), py::arg("tol") = 1e-8, py::arg("max_iter") = 500,
      py::arg("clear_assemblers") = true, py::arg("verbose") = true,
      py::call_guard<py::gil_scoped_release>())
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
    .def("get_assemblers", &TransportDriver::get_assemblers,
      "Return GID -> this rank's own local patch assembler, for every "
      "patch whose assembler hasn't been cleared. Pass to "
      "TransportSolution.compute_patch_balances()/patch_balance_table()/"
      "global_balance() (requires clear_assemblers=False when solving).",
      py::call_guard<py::gil_scoped_release>())
    .def(
      "set_callback",
      [](TransportDriver& self, typename TransportDriver::Callback callback,
        int frequency) {
        self.set_callback(
          [callback = std::move(callback)](const TransportDriver& driver,
            const ttnte::solvers::Solver& solver) {
            py::gil_scoped_acquire acquire;
            callback(driver, solver);
          },
          frequency);
      },
      py::arg("callback"), py::arg("frequency") = 1,
      "Set a callback invoked every `frequency`-th outer iteration inside "
      "solve_eigenvalue()/solve_fixed_source() (1 = every iteration, the "
      "default), with this driver and the inner_solver passed to that "
      "call. The frequency check happens before the GIL is reacquired to "
      "call back into Python, so a skipped iteration costs nothing. Pass "
      "None to disable.")

    // =================================================================
    // Public getters / setters
    .def_property(
      "label", &TransportDriver::get_label, &TransportDriver::set_label)
    .def_property_readonly("mesh", &TransportDriver::get_mesh)
    .def_property_readonly("server", &TransportDriver::get_server)
    .def_property_readonly("gid2rank", &TransportDriver::get_gid2rank)
    .def("last_k", &TransportDriver::last_k)
    .def(
      "last_num_outer_iterations", &TransportDriver::last_num_outer_iterations)
    .def("last_outer_error", &TransportDriver::last_outer_error);
}

void register_TransportDriver(py::module_& m)
{
  register_TransportDriver_impl<ttnte::cad::Patch, 1>(m, "IGA");
  register_TransportDriver_impl<ttnte::cad::Patch, 2>(m, "IGA");
  register_TransportDriver_impl<ttnte::cad::Patch, 3>(m, "IGA");
}
