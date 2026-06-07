#include "ttnte/physics/dg_first_order_transport_backends.hpp"
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

template<ttnte::physics::FormatType Fmt, int64_t NumDim>
void register_DGFirstOrderTransportBackend(
  py::module_& m, const std::string& name)
{
  using Patch = ttnte::cad::Patch;
  using Config = ttnte::physics::DGTransportAssemblerConfig;
  using Backend =
    ttnte::physics::backends::DGFirstOrderTransportBackend<ttnte::cad::Patch,
      Fmt, NumDim>;
  using Return = ttnte::physics::backends::Return<Fmt, NumDim>;
  using namespace ttnte::physics;

  py::class_<Backend>(m, name.c_str())
    // =================================================================
    // Public constructors
    .def(py::init<const Patch::Ptr&, const ttnte::math::QuadratureSet::Ptr&,
           const ttnte::xs::Server::Ptr&, const Config&>(),
      py::arg("patch"), py::arg("angular_qset"), py::arg("xs_server"),
      py::arg("config") = Config(), py::keep_alive<1, 3>())

    // =================================================================
    // Public methods
    .def(
      "assemble_ordinates",
      [](Backend& self) {
        if constexpr (Fmt == ttnte::physics::FormatType::DENSE) {
          return self.assemble_ordinates();

        } else if constexpr (Fmt == ttnte::physics::FormatType::TENSOR_TRAIN) {
          auto result = self.assemble_ordinates();
          return std::vector<typename Return::Type>(
            result.begin(), result.end());
        }
      },
      py::call_guard<py::gil_scoped_release>())
    .def("assemble_basis", &Backend::assemble_basis,
      py::call_guard<py::gil_scoped_release>())
    .def(
      "assemble_basis_ders",
      [](Backend& self) {
        if constexpr (Fmt == ttnte::physics::FormatType::DENSE) {
          return self.assemble_basis_ders();

        } else if constexpr (Fmt == ttnte::physics::FormatType::TENSOR_TRAIN) {
          auto result = self.assemble_basis_ders();
          return std::vector<typename Return::Type>(
            result.begin(), result.end());
        }
      },
      py::call_guard<py::gil_scoped_release>())
    .def("assemble_scattering_kernel", &Backend::assemble_scattering_kernel,
      py::call_guard<py::gil_scoped_release>(), py::arg("spatial") = py::none())
    .def("assemble_angular_integral", &Backend::assemble_angular_integral,
      py::call_guard<py::gil_scoped_release>())
    .def("assemble_jacobian", &Backend::assemble_jacobian,
      py::call_guard<py::gil_scoped_release>())
    .def("assemble_integral_mapping", &Backend::assemble_integral_mapping,
      py::call_guard<py::gil_scoped_release>())
    .def("assemble_jacobian_inverse", &Backend::assemble_jacobian_inverse,
      py::call_guard<py::gil_scoped_release>())
    .def("assemble_loss_operator", &Backend::assemble_loss_operator,
      py::call_guard<py::gil_scoped_release>())
    .def("assemble_scatter_operator", &Backend::assemble_scatter_operator,
      py::call_guard<py::gil_scoped_release>())
    .def("assemble_fission_operator", &Backend::assemble_fission_operator,
      py::call_guard<py::gil_scoped_release>())
    .def("assemble_boundary_operators", &Backend::assemble_boundary_operators,
      py::call_guard<py::gil_scoped_release>())
    .def(
      "apply_angular_weights",
      [](const Backend& self, const ttnte::linalg::TTEngine& op,
        const std::vector<size_t>& core_idxs) {
        return self.apply_angular_weights(op,
          c10::SmallVector<size_t, 2>(core_idxs.cbegin(), core_idxs.cend()));
      },
      py::call_guard<py::gil_scoped_release>())

    .def("to_", py::overload_cast<const torch::TensorOptions&>(&Backend::to_),
      py::arg("options"), py::call_guard<py::gil_scoped_release>())
    .def("to_",
      py::overload_cast<const torch::Device&, const torch::ScalarType&>(
        &Backend::to_),
      py::arg("device"), py::arg("dtype"),
      py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Public Getters / Setters
    .def("get_config", &Backend::get_config,
      py::return_value_policy::reference_internal)
    .def("get_block", &Backend::get_block,
      py::return_value_policy::reference_internal)
    .def("get_spatial_qset", &Backend::get_spatial_qset,
      py::return_value_policy::reference_internal)
    .def("get_angular_qset", &Backend::get_angular_qset,
      py::return_value_policy::reference_internal)
    .def("get_material", &Backend::get_material,
      py::return_value_policy::reference_internal)
    .def("get_quad_points",
      [](const Backend& self) {
        const auto& p = self.get_quad_points();
        return std::vector<torch::Tensor>(p.begin(), p.end());
      })

    .def_property_readonly("config", &Backend::get_config,
      py::return_value_policy::reference_internal)
    .def_property_readonly(
      "block", &Backend::get_block, py::return_value_policy::reference_internal)
    .def_property_readonly("spatial_qset", &Backend::get_spatial_qset,
      py::return_value_policy::reference_internal)
    .def_property_readonly("angular_qset", &Backend::get_angular_qset,
      py::return_value_policy::reference_internal)
    .def_property_readonly("material", &Backend::get_material,
      py::return_value_policy::reference_internal)
    .def_property_readonly("quad_points", [](const Backend& self) {
      const auto& p = self.get_quad_points();
      return std::vector<torch::Tensor>(p.begin(), p.end());
    });
}

void register_dg_first_order_transport_backends(py::module_& m)
{
  using ttnte::physics::FormatType;

  // Dense registers
  register_DGFirstOrderTransportBackend<FormatType::DENSE, 1>(
    m, "DenseDIGAFirstOrderTransportBackend1D");
  register_DGFirstOrderTransportBackend<FormatType::DENSE, 2>(
    m, "DenseDIGAFirstOrderTransportBackend2D");
  register_DGFirstOrderTransportBackend<FormatType::DENSE, 3>(
    m, "DenseDIGAFirstOrderTransportBackend3D");

  // TT registers
  register_DGFirstOrderTransportBackend<FormatType::TENSOR_TRAIN, 1>(
    m, "TTDIGAFirstOrderTransportBackend1D");
  register_DGFirstOrderTransportBackend<FormatType::TENSOR_TRAIN, 2>(
    m, "TTDIGAFirstOrderTransportBackend2D");
  register_DGFirstOrderTransportBackend<FormatType::TENSOR_TRAIN, 3>(
    m, "TTDIGAFirstOrderTransportBackend3D");
}
