#pragma once

#include "ttnte/parallel/heuristics.hpp"
#include <torch/extension.h>

namespace py = pybind11;

template<typename T>
class PyLoadHeuristic : public ttnte::parallel::heuristics::LoadHeuristic<T> {
public:
  // Inherit the constructors
  using LoadHeuristic =
    ttnte::parallel::heuristics::LoadHeuristic<T>::LoadHeuristic;
  using MeshBlock = ttnte::mesh::MeshBlock<T>;

  // Trampoline for compute_weights
  torch::Tensor compute_weights() const override
  {
    PYBIND11_OVERRIDE_PURE(torch::Tensor, // Return type
      LoadHeuristic,                      // Parent class
      compute_weights,                    // Name of function in C++
    );
  }
};

template<typename T>
void register_heuristics(py::module_& m, const std::string& typestr)
{
  using namespace ttnte::parallel::heuristics;
  using LoadHeuristic = LoadHeuristic<T>;
  using DofHeuristic = DofHeuristic<T>;

  // Bind the interface LoadHeuristic class
  std::string base_name = typestr + "LoadHeuristic";
  py::class_<LoadHeuristic, PyLoadHeuristic<T>, std::shared_ptr<LoadHeuristic>>(
    m, base_name.c_str())
    .def(py::init<>())
    .def("compute_weights", &LoadHeuristic::compute_weights);

  // Bind the concrete DofHeuristic class
  std::string dof_name = typestr + "DofHeuristic";
  py::class_<DofHeuristic, LoadHeuristic, std::shared_ptr<DofHeuristic>>(
    m, dof_name.c_str())
    .def(py::init<>());
}
