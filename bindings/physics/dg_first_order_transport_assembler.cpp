#include "ttnte/physics/dg_first_order_transport_assembler.hpp"
#include <torch/extension.h>

namespace py = pybind11;

template<typename BlockType, int64_t NumDim>
void register_DGFirstOrderTransportAssembler(
  py::module_& m, const std::string& class_name)
{
  using namespace ttnte;
  using namespace ttnte::physics;

  // Assuming BlockType is cad::Patch based on your typical usage
  using BaseType = DGAssembler<cad::Patch, DGTransportAssemblerConfig>;
  using AssemblerType = DGFirstOrderTransportAssembler<BlockType, NumDim>;

  py::class_<AssemblerType, BaseType, std::shared_ptr<AssemblerType>>(
    m, class_name.c_str())

    // ---------------------------------------------------------
    // Constructors (Using a lambda to wrap the perfect-forwarding factory)
    // ---------------------------------------------------------
    .def(py::init([](const cad::Patch::Ptr& block,
                    const math::QuadratureSet::Ptr& angular_qset,
                    const xs::Server::Ptr& xs_server,
                    const DGTransportAssemblerConfig& config) {
      return AssemblerType::create(block, angular_qset, xs_server, config);
    }),
      py::arg("block"), py::arg("angular_qset"), py::arg("xs_server"),
      py::arg("config") = DGTransportAssemblerConfig(),
      "Create a new DGFirstOrderTransportAssembler.")

    // ---------------------------------------------------------
    // Single Operator / Object Getters
    // ---------------------------------------------------------
    .def("get_interior_loss_op", &AssemblerType::get_interior_loss_op,
      py::return_value_policy::reference_internal)
    .def("get_scatter_op", &AssemblerType::get_scatter_op,
      py::return_value_policy::reference_internal)
    .def("get_fission_op", &AssemblerType::get_fission_op,
      py::return_value_policy::reference_internal)
    .def("get_source", &AssemblerType::get_source,
      py::return_value_policy::reference_internal)
    .def("get_angular_qset", &AssemblerType::get_angular_qset,
      py::return_value_policy::reference_internal)
    .def("get_xs_server", &AssemblerType::get_xs_server,
      py::return_value_policy::reference_internal)

    .def_property_readonly(
      "interior_loss_op", &AssemblerType::get_interior_loss_op)
    .def_property_readonly("scatter_op", &AssemblerType::get_scatter_op)
    .def_property_readonly("fission_op", &AssemblerType::get_fission_op)
    .def_property_readonly("source", &AssemblerType::get_source)
    .def_property_readonly("angular_qset", &AssemblerType::get_angular_qset)
    .def_property_readonly("xs_server", &AssemblerType::get_xs_server)

    // ---------------------------------------------------------
    // Vector Getters (Translating c10::SmallVector -> std::vector)
    // ---------------------------------------------------------
    .def(
      "get_outflow_ops",
      [](const AssemblerType& self) {
        const auto& ops = self.get_outflow_ops();
        return std::vector<ttnte::linalg::Operator>(ops.begin(), ops.end());
      },
      "Get the outflow boundary operators as a Python list.")
    .def(
      "get_inflow_ops",
      [](const AssemblerType& self) {
        const auto& ops = self.get_inflow_ops();
        return std::vector<ttnte::linalg::Operator>(ops.begin(), ops.end());
      },
      "Get the inflow boundary operators as a Python list.")
    .def_property_readonly(
      "outflow_ops",
      [](const AssemblerType& self) {
        const auto& ops = self.get_outflow_ops();
        return std::vector<ttnte::linalg::Operator>(ops.begin(), ops.end());
      },
      "Get the outflow boundary operators as a Python list.")
    .def_property_readonly(
      "inflow_ops",
      [](const AssemblerType& self) {
        const auto& ops = self.get_inflow_ops();
        return std::vector<ttnte::linalg::Operator>(ops.begin(), ops.end());
      },
      "Get the inflow boundary operators as a Python list.");

  // Note: We intentionally skip binding get_backend_variant and get_backend
  // to Python. Exposing raw pointers to internally managed unique_ptrs
  // via std::variant is extremely dangerous for Python's garbage collector.
}

void register_dg_first_order_transport_assemblers(py::module_& m)
{
  // Explicitly instantiate for the dimensions you use in Python
  register_DGFirstOrderTransportAssembler<ttnte::cad::Patch, 1>(
    m, "DIGAFirstOrderTransportAssembler1D");
  register_DGFirstOrderTransportAssembler<ttnte::cad::Patch, 2>(
    m, "DIGAFirstOrderTransportAssembler2D");
  register_DGFirstOrderTransportAssembler<ttnte::cad::Patch, 3>(
    m, "DIGAFirstOrderTransportAssembler3D");
}
