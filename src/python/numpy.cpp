#include "ttnte/python/numpy.hpp"
#include "ttnte/python/package_manager.hpp"
#include <pybind11/numpy.h>

namespace ttnte::python::numpy {

Acquire::Acquire() : python::Acquire() {}
Acquire::~Acquire() = default;

std::pair<torch::Tensor, torch::Tensor> Acquire::leggauss(int64_t deg) const
{
  // Run numpy leggauss function
  py::tuple result = PackageManager::instance()
                       .numpy.attr("polynomial")
                       .attr("legendre")
                       .attr("leggauss")(deg);

  // Get the points and weights
  auto points_np = result[0].cast<py::array_t<double>>();
  auto weights_np = result[1].cast<py::array_t<double>>();

  // Convert to libtorch
  torch::Tensor points =
    torch::from_blob(points_np.mutable_data(), {deg}, torch::kFloat64).clone();
  torch::Tensor weights =
    torch::from_blob(weights_np.mutable_data(), {deg}, torch::kFloat64).clone();

  return {points, weights};
}

} // namespace ttnte::python::numpy
