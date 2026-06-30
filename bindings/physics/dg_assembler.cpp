#include "ttnte/physics/dg_assembler.hpp"
#include "ttnte/cad/patch.hpp"
#include <torch/extension.h>

namespace py = pybind11;

template<typename BlockType, typename ConfigType>
void register_DGAssembler(py::module_& m, const std::string& class_name)
{
  using namespace ttnte::physics;
  using BaseType = DGAssembler<BlockType, ConfigType>;

  // Abstract class: No constructors are bound!
  py::class_<BaseType, std::shared_ptr<BaseType>>(m, class_name.c_str())
    .def("assemble", &BaseType::assemble,
      "Assemble the linear system and return the LinearSystem pointer.")
    .def("get_config", &BaseType::get_config,
      "Get the configuration for assembly.")
    .def(
      "get_block", &BaseType::get_block, "Get the pointer to the mesh block.")
    .def("get_linear_system", &BaseType::get_linear_system,
      "Get the assembled linear system pointer.");
}

void register_dg_assemblers(py::module_& m)
{
  register_DGAssembler<ttnte::cad::Patch,
    ttnte::physics::DGTransportAssemblerConfig>(m, "DIGAAssembler");
}
