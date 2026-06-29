#include "ttnte/solvers/solver_configs.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_DDSolverConfig(py::module_& m)
{
  using namespace ttnte::solvers;
  using namespace ttnte::linalg;

  py::enum_<ExecMode>(m, "ExecMode")
    .value("SYNC", ExecMode::SYNC)
    .value("ASYNC", ExecMode::ASYNC)
    .export_values();

  py::enum_<CommMode>(m, "CommMode")
    .value("SYNC", CommMode::SYNC)
    .value("ASYNC", CommMode::ASYNC)
    .export_values();

  py::class_<DDSolverConfig>(m, "DDSolverConfig")
    // Flat constructor — all parameters with defaults
    .def(py::init<double, int, FormatType, double, int64_t, bool, ExecMode,
           CommMode, int, int, bool, MemoryPolicy, double, double, bool>(),
      py::arg("tol") = 1e-8, py::arg("max_iter") = 100,
      py::arg("fmt") = FormatType::TENSOR_TRAIN, py::arg("eps") = 1e-12,
      py::arg("max_rank") = 500, py::arg("clear_assemblers") = true,
      py::arg("exec_mode") = ExecMode::ASYNC,
      py::arg("comm_mode") = CommMode::ASYNC, py::arg("num_threads") = 4,
      py::arg("num_streams") = 16, py::arg("use_gpu") = DEFAULT_USE_GPU,
      py::arg("memory_policy") = DEFAULT_MEMORY_POLICY,
      py::arg("inner_forcing") = 0.1, py::arg("eps_forcing") = 0.01,
      py::arg("verbose") = false)
    // Rounding-first constructor
    .def(py::init<TTConfig, double, int, FormatType, bool, ExecMode, CommMode,
           int, int, bool, MemoryPolicy, double, double, bool>(),
      py::arg("rounding"), py::arg("tol") = 1e-8, py::arg("max_iter") = 100,
      py::arg("fmt") = FormatType::TENSOR_TRAIN,
      py::arg("clear_assemblers") = true,
      py::arg("exec_mode") = ExecMode::ASYNC,
      py::arg("comm_mode") = CommMode::ASYNC, py::arg("num_threads") = 4,
      py::arg("num_streams") = 16, py::arg("use_gpu") = DEFAULT_USE_GPU,
      py::arg("memory_policy") = DEFAULT_MEMORY_POLICY,
      py::arg("inner_forcing") = 0.1, py::arg("eps_forcing") = 0.01,
      py::arg("verbose") = false)

    // =================================================================
    // Fields
    .def_readwrite("rounding", &DDSolverConfig::rounding)
    .def_readwrite("fmt", &DDSolverConfig::fmt)
    .def_readwrite("tol", &DDSolverConfig::tol)
    .def_readwrite("max_iter", &DDSolverConfig::max_iter)
    .def_readwrite("clear_assemblers", &DDSolverConfig::clear_assemblers)
    .def_readwrite("exec_mode", &DDSolverConfig::exec_mode)
    .def_readwrite("comm_mode", &DDSolverConfig::comm_mode)
    .def_readwrite("num_threads", &DDSolverConfig::num_threads)
    .def_readwrite("num_streams", &DDSolverConfig::num_streams)
    .def_readwrite("use_gpu", &DDSolverConfig::use_gpu)
    .def_readwrite("memory_policy", &DDSolverConfig::memory_policy)
    .def_readwrite("verbose", &DDSolverConfig::verbose)
    .def_readwrite("inner_forcing", &DDSolverConfig::inner_forcing)
    .def_readwrite("eps_forcing", &DDSolverConfig::eps_forcing);
}
