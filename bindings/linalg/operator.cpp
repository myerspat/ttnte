#include "ttnte/linalg/operator.hpp"
#include "../utils/label.hpp"
#include <torch/extension.h>

namespace py = pybind11;

class PyOperator : public ttnte::linalg::Operator {
public:
  using Operator::Operator; // Inherit constructors

  // Trampoline for pure virtual to_impl
  Ptr to_impl(const torch::Device& device, const at::ScalarType& dtype,
    bool non_blocking, bool copy,
    std::optional<at::MemoryFormat> memory_format) const override
  {
    PYBIND11_OVERRIDE_PURE(
      Ptr, Operator, to_impl, device, dtype, non_blocking, copy, memory_format);
  }

  // Trampoline for pure virtual get_device
  torch::Device get_device() const override
  {
    PYBIND11_OVERRIDE_PURE(torch::Device, Operator, get_device);
  }

  // Trampoline for pure virtual get_dtype
  at::ScalarType get_dtype() const override
  {
    PYBIND11_OVERRIDE_PURE(at::ScalarType, Operator, get_dtype);
  }

  // Trampoline for pure virtual get_numel
  int64_t get_numel() const override
  {
    PYBIND11_OVERRIDE_PURE(int64_t, Operator, get_numel);
  }
};

void register_Operator(py::module_& m)
{
  using namespace ttnte::linalg;
  register_Label<Operator>(m, "Operator");

  py::class_<Operator, PyOperator, Operator::Ptr>(m, "Operator")
    // =================================================================
    // Public methods
    .def("is_cuda", &Operator::is_cuda)
    .def("to",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to, py::const_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to, py::const_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to, py::const_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("to_",
      py::overload_cast<const torch::Device&, const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to_),
      py::arg("device"), py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const at::ScalarType&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to_),
      py::arg("dtype"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())
    .def("to_",
      py::overload_cast<const torch::Device&, bool, bool,
        std::optional<at::MemoryFormat>>(&Operator::to_),
      py::arg("device"), py::arg("non_blocking") = false,
      py::arg("copy") = false, py::arg("memory_format") = py::none())

    .def("get_label", &Operator::get_label,
      py::return_value_policy::reference_internal)

    // =================================================================
    // Public getters / setters
    .def_property_readonly("label", &Operator::get_label)
    .def_property_readonly("device", &Operator::get_device)
    .def_property_readonly("dtype", &Operator::get_dtype)
    .def_property_readonly("numel", &Operator::get_numel);
}
