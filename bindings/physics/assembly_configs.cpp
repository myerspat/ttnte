#include "ttnte/physics/assembly_configs.hpp"
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_assembly_configs(py::module_& m)
{
  using namespace ttnte::physics;

  // =================================================================
  // FormatType Enum
  py::enum_<FormatType>(m, "FormatType")
    .value("DENSE", FormatType::DENSE)
    .value("TENSOR_TRAIN", FormatType::TENSOR_TRAIN)
    .export_values();

  // =================================================================
  // TTConfig
  py::class_<TTConfig>(m, "TTConfig")
    .def(py::init<double, int>(), py::arg("eps") = 1e-12,
      py::arg("max_rank") = 500)
    .def_readwrite("eps", &TTConfig::eps)
    .def_readwrite("max_rank", &TTConfig::max_rank);

  // =================================================================
  // CrossConfig (Inherits from TTConfig)
  py::class_<CrossConfig, TTConfig>(m, "CrossConfig")
    .def(py::init<double, int, int, int, bool>(), py::arg("eps") = 1e-12,
      py::arg("max_rank") = 500, py::arg("nswp") = 50, py::arg("kick") = 4,
      py::arg("verbose") = false)
    .def_readwrite("nswp", &CrossConfig::nswp)
    .def_readwrite("kick", &CrossConfig::kick)
    .def_readwrite("verbose", &CrossConfig::verbose);

  // =================================================================
  // DGAssemblerConfig
  py::class_<DGAssemblerConfig>(m, "DGAssemblerConfig")
    .def(
      py::init<double, int, int, int, int64_t, bool,
        std::optional<torch::Device>, std::optional<torch::ScalarType>, bool>(),
      py::arg("eps") = 1e-12, py::arg("max_rank") = 500, py::arg("nswp") = 50,
      py::arg("kick") = 4, py::arg("max_dense_size") = 100000,
      py::arg("cross_jacobian_inverse") = false, py::arg("device") = py::none(),
      py::arg("dtype") = py::none(), py::arg("verbose") = false)
    .def(py::init<const TTConfig&, const CrossConfig&, int64_t, bool,
           std::optional<torch::Device>, std::optional<torch::ScalarType>>(),
      py::arg("rounding"), py::arg("cross"), py::arg("max_dense_size") = 100000,
      py::arg("cross_jacobian_inverse") = false, py::arg("device") = py::none(),
      py::arg("dtype") = py::none())
    .def_readwrite("device", &DGAssemblerConfig::device)
    .def_readwrite("dtype", &DGAssemblerConfig::dtype)
    .def_readwrite("max_dense_size", &DGAssemblerConfig::max_dense_size)
    .def_readwrite(
      "cross_jacobian_inverse", &DGAssemblerConfig::cross_jacobian_inverse)
    .def_readwrite("rounding", &DGAssemblerConfig::rounding)
    .def_readwrite("cross", &DGAssemblerConfig::cross);

  // =================================================================
  // DGTransportAssemblerConfig (Inherits from DGAssemblerConfig)
  py::class_<DGTransportAssemblerConfig, DGAssemblerConfig>(
    m, "DGTransportAssemblerConfig")
    .def(py::init<>())
    .def_readwrite(
      "interior_loss_fmt", &DGTransportAssemblerConfig::interior_loss_fmt)
    .def_readwrite("scatter_fmt", &DGTransportAssemblerConfig::scatter_fmt)
    .def_readwrite("source_fmt", &DGTransportAssemblerConfig::source_fmt)
    .def_readwrite("outflow_fmt", &DGTransportAssemblerConfig::outflow_fmt)
    .def_readwrite("inflow_fmt", &DGTransportAssemblerConfig::inflow_fmt);
}
