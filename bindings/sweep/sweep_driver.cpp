#include "ttnte/sweep/sweep_driver.hpp"
#include "../cross/history.hpp"
#include "ttnte/cad/patch.hpp"
#include "ttnte/cross/layout_cross.hpp"
#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/sampling/fiber_store.hpp"
#include "ttnte/sweep/projected_transport.hpp"
#include <algorithm>
#include <optional>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <torch/extension.h>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace {

using ttnte::sweep::SweepSolver;

/// Deprecated read-only alias `old_name` of a record (warns, then returns
/// get(solver)): the cross_* records are the samplers' since GALERKIN.
template<typename Get>
void deprecated_record(py::class_<SweepSolver, SweepSolver::Ptr>& cls,
  const char* old_name, const char* use, Get get)
{
  const std::string message = std::string("SweepDriver.") + old_name +
                              " is deprecated; use " + use +
                              " (shared by CROSS and GALERKIN)";
  cls.def_property_readonly(
    old_name,
    [get, message](const SweepSolver& self) {
      if (PyErr_WarnEx(PyExc_FutureWarning, message.c_str(), 1) < 0)
        throw py::error_already_set();
      return get(self);
    },
    ("Deprecated: use " + std::string(use) + ".").c_str());
}

/// One field of every sampled transport solve's record.
template<typename F>
auto per_record(const SweepSolver& self, F field)
{
  using T = std::decay_t<decltype(field(*self.get_sampling_records()[0]))>;
  std::vector<T> out;
  for (const auto& r : self.get_sampling_records())
    out.push_back(field(*r));
  return out;
}

} // namespace

template<typename BlockType, int64_t NumDim>
static void register_SweepDriver_impl(
  py::module_& m, const std::string& typestr)
{
  using SweepDriver = ttnte::sweep::SweepDriver<BlockType, NumDim>;
  std::string class_name =
    typestr + "SweepDriver" + std::to_string(NumDim) + "D";

  py::class_<SweepDriver, ttnte::sweep::SweepSolver, typename SweepDriver::Ptr>(
    m, class_name.c_str())
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
    .def(
      "sweep_fibers",
      [](SweepDriver& self,
        const std::vector<
          std::pair<std::vector<int64_t>, std::vector<int64_t>>>& blocks) {
        std::vector<typename SweepDriver::FiberBlock> request;
        int64_t num_fibers = 0;
        for (const auto& [dirs, groups] : blocks) {
          request.push_back({dirs, groups});
          num_fibers += static_cast<int64_t>(dirs.size() * groups.size());
        }
        auto out =
          torch::empty({num_fibers, self.fiber_length()}, torch::kFloat64);
        {
          py::gil_scoped_release release;
          self.sweep_fibers(request, out.template data_ptr<double>());
        }
        return out;
      },
      "Sweep (directions, groups) blocks of fibers with the current sources "
      "and lagged traces. Returns (num_fibers, fiber_length): block by "
      "block, direction-major then group; each row holds the local units' "
      "dofs in local order (see fiber_offsets).",
      py::arg("blocks"))
    .def(
      "transport_residual",
      [](const SweepDriver& self, int64_t direction,
        const std::vector<int64_t>& groups, const torch::Tensor& psi) {
        const int64_t ng = static_cast<int64_t>(groups.size());
        if (psi.dim() != 2 || psi.size(0) != ng ||
            psi.size(1) != self.fiber_length())
          throw py::value_error("psi must be (len(groups), fiber_length)");
        auto x = psi.to(torch::kFloat64).contiguous();
        auto r = torch::empty_like(x);
        {
          py::gil_scoped_release release;
          self.transport_residual(direction, groups,
            x.template data_ptr<double>(), r.template data_ptr<double>());
        }
        return r;
      },
      "Transport residual source + inflow - L psi of a candidate flux psi "
      "(len(groups), fiber_length) of one direction, with the current "
      "sources and lagged traces (those sweep_fibers reads); a swept fiber's "
      "is at rounding level (testing aid).",
      py::arg("direction"), py::arg("groups"), py::arg("psi"))
    .def(
      "projected_system",
      [](const SweepDriver& self, const std::vector<int64_t>& directions,
        const std::vector<torch::Tensor>& chunks, size_t group_set,
        bool flat_faces, bool assemble_each,
        const std::vector<torch::Tensor>& previous, bool implicit_lagged,
        bool sign_split, int64_t max_face_classes, double max_class_ratio,
        bool spatial_basis) {
        using ttnte::sweep::ProjectedTransport;
        auto pt = ProjectedTransport::create(self, group_set,
          ProjectedTransport::Options {flat_faces, implicit_lagged, sign_split,
            max_face_classes, max_class_ratio, spatial_basis});
        // An earlier basis, every direction assembled on it, then dropped:
        // what a sampler does between transport solves
        if (!previous.empty()) {
          for (const auto& X : previous)
            pt->extend(X.to(torch::kFloat64).contiguous());
          for (int64_t d : directions)
            pt->reduced_system(d);
          pt->reset();
        }
        for (const auto& X : chunks) {
          pt->extend(X.to(torch::kFloat64).contiguous());
          // Assembled between extensions: the curved-face caches are then
          // bordered lazily, chunk by chunk
          for (int64_t d : directions)
            if (assemble_each)
              pt->reduced_system(d);
        }
        std::vector<torch::Tensor> A, b;
        for (int64_t d : directions) {
          auto [Ad, bd] = pt->reduced_system(d);
          A.push_back(Ad);
          b.push_back(bd);
        }
        return py::make_tuple(torch::stack(A), torch::stack(b),
          pt->num_normals(), pt->num_curved_faces(), pt->face_classes());
      },
      "Reduced systems X^T A_d X, X^T b_d of directions on a basis given in "
      "chunks (each (p, |set|, fiber_length), bordered in turn), with the "
      "current sources and lagged traces: (A (D, r, r), b (D, r), distinct "
      "flat normals, curved faces, classes per curved face). flat_faces=False "
      "projects every face by "
      "quadrature; assemble_each=True assembles the directions after every "
      "chunk; previous: a basis projected (every direction assembled) and "
      "reset() before the chunks; implicit_lagged: lagged edges in A, not b; "
      "sign_split / max_face_classes / max_class_ratio: curved faces per "
      "sign class (the last returned value: classes per curved face, -1 per "
      "direction); spatial_basis: pair sampling, `directions` are then units "
      "d |set| + g and the chunks (p, 1, fiber_length) (testing aid).",
      py::arg("directions"), py::arg("chunks"), py::arg("group_set") = 0,
      py::arg("flat_faces") = true, py::arg("assemble_each") = false,
      py::arg("previous") = std::vector<torch::Tensor> {},
      py::arg("implicit_lagged") = false, py::arg("sign_split") = true,
      py::arg("max_face_classes") = 64, py::arg("max_class_ratio") = 1.0,
      py::arg("spatial_basis") = false)
    .def(
      "buffered_projection",
      [](const SweepDriver& self, const std::vector<int64_t>& directions,
        const torch::Tensor& X, size_t group_set) {
        using ttnte::sweep::ProjectedTransport;
        auto pt = ProjectedTransport::create(
          self, group_set, ProjectedTransport::Options {});
        pt->extend(X.to(torch::kFloat64).contiguous());
        std::vector<torch::Tensor> b;
        for (int64_t d : directions)
          b.push_back(pt->buffered_rhs(d));
        return torch::stack(b);
      },
      "The buffered part of the reduced right-hand sides, X^T (lagged and "
      "reflected inflow of d), (D, r) (testing aid).",
      py::arg("directions"), py::arg("X"), py::arg("group_set") = 0)
    .def(
      "galerkin_solve",
      [](const SweepDriver& self, const std::vector<int64_t>& directions,
        const std::vector<torch::Tensor>& chunks, size_t group_set,
        bool spatial_basis) {
        using ttnte::sweep::ProjectedTransport;
        auto pt = ProjectedTransport::create(self, group_set,
          ProjectedTransport::Options {.spatial_basis = spatial_basis});
        for (const auto& X : chunks)
          pt->extend(X.to(torch::kFloat64).contiguous());
        return pt->solve(directions);
      },
      "Reduced coefficients C (D, r) of directions (units with "
      "spatial_basis) on a basis given in chunks (ProjectedTransport::solve, "
      "parallel over units on torch's threads; testing aid).",
      py::arg("directions"), py::arg("chunks"), py::arg("group_set") = 0,
      py::arg("spatial_basis") = false)
    .def(
      "galerkin_residuals",
      [](const SweepDriver& self, const std::vector<int64_t>& directions,
        const std::vector<torch::Tensor>& chunks, const torch::Tensor& C,
        size_t group_set, bool spatial_basis) {
        using ttnte::sweep::ProjectedTransport;
        auto pt = ProjectedTransport::create(self, group_set,
          ProjectedTransport::Options {.spatial_basis = spatial_basis});
        for (const auto& X : chunks)
          pt->extend(X.to(torch::kFloat64).contiguous());
        return pt->residuals(directions, C);
      },
      "Residuals b_d - A_d X c_d (D, |set|, fiber_length) of coefficients C "
      "(D, r) on a basis given in chunks; with spatial_basis per unit d |set| "
      "+ g, (U, 1, fiber_length) (testing aid).",
      py::arg("directions"), py::arg("chunks"), py::arg("C"),
      py::arg("group_set") = 0, py::arg("spatial_basis") = false)
    .def(
      "galerkin_scattering",
      [](const SweepDriver& self, const std::vector<torch::Tensor>& chunks,
        size_t group_set, std::optional<torch::Tensor> phi) {
        using ttnte::sweep::ProjectedTransport;
        auto pt = ProjectedTransport::create(
          self, group_set, ProjectedTransport::Options {.scattering = true});
        for (const auto& X : chunks)
          pt->extend(X.to(torch::kFloat64).contiguous());
        std::vector<torch::Tensor> K;
        for (int64_t l = 0; l < pt->num_orders(); l++)
          K.push_back(pt->scattering(l));
        const auto moments = phi ? *phi : pt->solver_moments();
        return py::make_tuple(torch::stack(K), pt->source_projection(),
          pt->project_scattering(moments), pt->solver_moments());
      },
      "Projected within-set scattering on a basis given in chunks: K (L, r, "
      "r) = X^T S_l X, the projected source X^T s (M, r), X^T S phi (M, r) "
      "of phi (M, |set|, fiber_length) (default: the solver's moments), and "
      "those moments (testing aid).",
      py::arg("chunks"), py::arg("group_set") = 0, py::arg("phi") = py::none())
    .def(
      "galerkin_angular",
      [](const SweepDriver& self, const std::vector<torch::Tensor>& chunks,
        std::optional<torch::Tensor> F, size_t group_set) {
        using ttnte::sweep::ProjectedTransport;
        auto pt = ProjectedTransport::create(
          self, group_set, ProjectedTransport::Options {.scattering = true});
        for (const auto& X : chunks)
          pt->extend(X.to(torch::kFloat64).contiguous());
        const auto f = F ? F->to(torch::kFloat64)
                         : pt->source_projection() -
                             pt->project_scattering(pt->solver_moments());
        auto [L, Lam] = pt->angular_galerkin(f);
        return py::make_tuple(L, Lam, f);
      },
      "Han et al.'s angular Galerkin problem on a basis given in chunks: L "
      "(D, r) and Lam (M, r) for the projected fixed source F (M, r) "
      "(default: X^T s - X^T S phi of the solver's source and moments), and "
      "that F (testing aid).",
      py::arg("chunks"), py::arg("F") = py::none(), py::arg("group_set") = 0)
    .def(
      "residual_norms",
      [](SweepDriver& self, const std::vector<int64_t>& directions,
        const torch::Tensor& residuals, size_t group_set,
        const ttnte::sampling::Weighting& weighting) {
        auto store = ttnte::sampling::FiberStore::create(
          self.sampling_problem(group_set), weighting);
        return py::make_tuple(
          store->residual_norms(directions, residuals), store->weighting());
      },
      "Dual weighted norms (D) of residuals (D, |set|, fiber_length) by a "
      "FiberStore of the set's problem, and the factors in effect (testing "
      "aid).",
      py::arg("directions"), py::arg("residuals"), py::arg("group_set") = 0,
      py::arg("weighting") = ttnte::sampling::Weighting {})
    .def_property_readonly("fiber_length", &SweepDriver::fiber_length)
    .def_property_readonly("fiber_offsets", &SweepDriver::fiber_offsets)
    .def_property_readonly("num_fibers", &SweepDriver::num_fibers)
    .def_property_readonly("local_gids", &SweepDriver::local_gids)
    .def_property_readonly(
      "num_kronecker_units", &SweepDriver::num_kronecker_units)
    .def_property_readonly("num_extruded_units",
      &SweepDriver::num_extruded_units,
      "Local units solved by the extruded-patch (partially separable) "
      "solver")
    .def_property_readonly("iterative_statistics",
      &SweepDriver::iterative_statistics,
      "(local solves, local GMRES applications, LU fallbacks) of the "
      "ITERATIVE solver")
    // The samplers reference the solver (sweep oracle, projected
    // transport): a sampler handed to Python keeps its driver alive
    .def("get_galerkin", &SweepDriver::get_galerkin, py::arg("group_set") = 0,
      py::keep_alive<0, 1>(),
      "The greedy Galerkin sampler of a group set (transport = GALERKIN): "
      "its last run's flux, history and records.")
    .def("get_cross", &SweepDriver::get_cross, py::arg("group_set") = 0,
      py::keep_alive<0, 1>(),
      "The LayoutCross of a group set (transport = CROSS)")
    .def("get_patch_angular_flux_tt", &SweepDriver::get_patch_angular_flux_tt,
      py::arg("gid"), py::arg("group_set") = 0,
      "Cores of a patch's angular flux TT (cross.keep_angular_flux) in the "
      "configured "
      "layout")
    .def_property_readonly("sampling_records",
      &SweepDriver::get_sampling_records,
      "Per sampled transport solve (CROSS, GALERKIN), its record: a RunRecord "
      "(tolerance, fibers, ranks, max_rank, error, converged, seconds, "
      "sweep_seconds) of the sampler's own subclass (CrossRun: history of "
      "half-sweeps; GalerkinRun: history of greedy iterations, "
      "dropped_snapshots, orthogonality_failures)")
    .def("angular_solution", &SweepDriver::angular_solution,
      "The solution with angular dependence: the stored angular flux, or with "
      "transport = CROSS and cross.keep_angular_flux the TT of the last "
      "transport solve")
    .def_property_readonly("sampling_seconds",
      &SweepDriver::get_sampling_seconds, "Wall seconds of every sampler run")
    .def_property_readonly("sampling_timings",
      &SweepDriver::get_sampling_timings,
      "Seconds per phase of the sampled transport solves (the samplers' "
      "phases summed over the group sets, and moments, traces, "
      "angular_flux)")
    .def_property_readonly("solve_timings", &SweepDriver::get_solve_timings,
      "Seconds per step of the within-group-set solves: gmres (whole Krylov "
      "solves), unpack, source, transport, pack.")
    .def_property_readonly("sampling_sweep_seconds",
      &SweepDriver::get_sampling_sweep_seconds,
      "Wall seconds of the samplers' fiber sweeps")
    .def("solve_fixed_source", &SweepDriver::solve_fixed_source,
      "Solve a fixed-source problem with the configured within-group-set "
      "solver (GMRES by default) and group-set iteration. "
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
          [callback = std::move(*callback)](
            const ttnte::sweep::SweepSolver& driver) {
            py::gil_scoped_acquire acquire;
            // The solver is this driver (its typed front end is registered)
            callback(static_cast<const SweepDriver&>(driver));
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
    .def_property_readonly("moment_order", &SweepDriver::get_moment_order,
      "(num_moments) Legendre order of each moment of the harmonics.")
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
    .def_property_readonly("incremental_solves",
      &SweepDriver::get_incremental_solves,
      "Correction-form y0 solves done by increment (incremental_transport).")
    .def_property_readonly("full_solves", &SweepDriver::get_full_solves,
      "Correction-form y0 solves done by a full transport solve "
      "(incremental_transport).")
    .def_property_readonly("equivalent_sweeps",
      &SweepDriver::get_equivalent_sweeps,
      "Full-sweep equivalents of the last solve (sum of |set| / G).")
    .def_property_readonly(
      "num_delayed_blocks", &SweepDriver::num_delayed_blocks)
    .def_property_readonly("converged", &SweepDriver::is_converged)
    .def_property_readonly("certification_error",
      &SweepDriver::get_certification_error,
      "SweepConfig.certify: scalar-flux change (relative L2) of the exact "
      "transport solve after the outer iteration; NaN without one.")
    .def_property_readonly("certification_k_change",
      &SweepDriver::get_certification_k_change,
      "SweepConfig.certify: change in k of that solve (eigenvalue only).")
    .def_property_readonly(
      "certification_seconds", &SweepDriver::get_certification_seconds)
    .def_property_readonly("num_tasks", &SweepDriver::num_tasks)
    .def_property_readonly("chunks", &SweepDriver::get_chunks)
    .def("memory_report", &SweepDriver::memory_report,
      "Bytes of the solver's large buffers on this rank, by kind (gmres is "
      "allocated during a solve).")
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
  // The dimension-independent core: the base of every IGASweepDriver*D, so
  // its methods bind once for all of them (not constructible from Python)
  auto solver = py::class_<SweepSolver, SweepSolver::Ptr>(m, "SweepSolver");

  // TODO: remove the cross_* records once the scripts (ttnte-vnv, studies)
  // read sampling_records, sampling_seconds, sampling_timings and
  // sampling_sweep_seconds
  using ttnte::sampling::RunRecord;
  deprecated_record(solver, "cross_tolerances",
    "[r.tolerance for r in sampling_records]", [](const SweepSolver& self) {
      return per_record(self, [](const RunRecord& r) { return r.tolerance; });
    });
  deprecated_record(solver, "cross_fibers",
    "[r.fibers for r in sampling_records]", [](const SweepSolver& self) {
      return per_record(self, [](const RunRecord& r) { return r.fibers; });
    });
  deprecated_record(solver, "cross_errors",
    "[r.error for r in sampling_records]", [](const SweepSolver& self) {
      return per_record(self, [](const RunRecord& r) { return r.error; });
    });
  deprecated_record(solver, "cross_ranks",
    "[r.ranks for r in sampling_records]", [](const SweepSolver& self) {
      return per_record(self, [](const RunRecord& r) { return r.ranks; });
    });
  deprecated_record(solver, "cross_max_ranks",
    "[r.max_rank for r in sampling_records]", [](const SweepSolver& self) {
      return per_record(self, [](const RunRecord& r) {
        int64_t out = 0;
        for (int64_t v : r.ranks)
          out = std::max(out, v);
        return out;
      });
    });
  deprecated_record(solver, "cross_histories",
    "[r.history for r in sampling_records]", [](const SweepSolver& self) {
      py::list out;
      for (const auto& r : self.get_sampling_records())
        if (const auto* c =
              dynamic_cast<const ttnte::cross::LayoutCross::RunRecord*>(
                r.get()))
          out.append(ttnte::bindings::sweep_history(c->history));
      return out;
    });
  deprecated_record(solver, "cross_seconds", "sampling_seconds",
    [](const SweepSolver& self) { return self.get_sampling_seconds(); });
  deprecated_record(solver, "cross_sweep_seconds", "sampling_sweep_seconds",
    [](const SweepSolver& self) { return self.get_sampling_sweep_seconds(); });
  deprecated_record(solver, "cross_timings", "sampling_timings",
    [](const SweepSolver& self) { return self.get_sampling_timings(); });

  register_SweepDriver_impl<ttnte::cad::Patch, 1>(m, "IGA");
  register_SweepDriver_impl<ttnte::cad::Patch, 2>(m, "IGA");
  register_SweepDriver_impl<ttnte::cad::Patch, 3>(m, "IGA");
}
