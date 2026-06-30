#include "ttnte/linalg/format_type.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_FormatType(py::module_& m)
{
  using namespace ttnte::linalg;

  py::enum_<FormatType>(m, "FormatType")
    .value("DENSE", FormatType::DENSE)
    .value("TENSOR_TRAIN", FormatType::TENSOR_TRAIN)
    .export_values();
}
