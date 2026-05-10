#include "ttnte/python/torchtt.hpp"
#include "ttnte/python/package_manager.hpp"
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace {

py::object ttnte2torchtt(const ttnte::linalg::TTEngine& engine)
{
  py::list cores;

  for (const auto& core : engine.get_cores()) {
    cores.append(py::cast(core));
  }

  return ttnte::python::PackageManager::instance().torchtt.attr("TT")(cores);
}

ttnte::linalg::TTEngine torchtt2ttnte(const py::object& py_engine)
{
  // Get the list of cores
  py::list py_cores = py_engine.attr("cores").cast<py::list>();

  ttnte::linalg::TTEngine::Tensors cores;
  for (auto core : py_cores) {
    cores.push_back(core.cast<torch::Tensor>());
  }

  return ttnte::linalg::TTEngine(cores, false);
}

} // namespace

namespace ttnte::python::torchtt {

Acquire::Acquire() : python::Acquire() {}
Acquire::~Acquire() = default;

linalg::TTEngine Acquire::amen_divide(const linalg::TTEngine& a,
  const linalg::TTEngine& b, int nswp, std::optional<linalg::TTEngine> x0,
  double eps, int rmax, int max_full, int kickrank, int kick2,
  std::string trunc_norm, int local_iterations, int resets, bool verbose,
  std::optional<std::string> preconditioner)
{
  const auto& a_cores = a.get_cores();
  const auto& b_cores = b.get_cores();
  assert(a_cores.size() == b_cores.size());
  size_t num_cores = a_cores.size();

  py::list py_a_cores;
  py::list py_b_cores;

  for (size_t i = 0; i < num_cores; i++) {
    const auto& a_core = a_cores[i];
    const auto& b_core = b_cores[i];

    py_a_cores.append(
      py::cast(a_core.reshape({a_core.size(0), -1, a_core.size(3)})));
    py_b_cores.append(
      py::cast(b_core.reshape({b_core.size(0), -1, b_core.size(3)})));
  }

  // Get the Python versions of the TTs
  py::object py_a =
    ttnte::python::PackageManager::instance().torchtt.attr("TT")(py_a_cores);
  py::object py_b =
    ttnte::python::PackageManager::instance().torchtt.attr("TT")(py_b_cores);

  // Handle the initial guess
  py::object py_x0 = py::none();
  if (x0.has_value()) {
    const auto& x0_cores = x0->get_cores();
    assert(a_cores.size() == x0_cores.size());

    py::list py_x0_cores;
    for (size_t i = 0; i < num_cores; i++) {
      const auto& x0_core = x0_cores[i];
      py_x0_cores.append(
        py::cast(x0_core.reshape({x0_core.size(0), -1, x0_core.size(3)})));
    }

    py_x0 =
      ttnte::python::PackageManager::instance().torchtt.attr("TT")(py_x0_cores);
  }

  // Call torchtt.amen_divide with all the parameters
  auto py_x_cores =
    PackageManager::instance()
      .torchtt.attr("_division")
      .attr("amen_divide")(py_b, py_a, py::arg("nswp") = nswp,
        py::arg("x0") = py_x0, py::arg("eps") = eps, py::arg("rmax") = rmax,
        py::arg("max_full") = max_full, py::arg("kickrank") = kickrank,
        py::arg("kick2") = kick2, py::arg("trunc_norm") = trunc_norm,
        py::arg("local_iterations") = local_iterations,
        py::arg("resets") = resets, py::arg("verbose") = verbose,
        py::arg("preconditioner") = preconditioner)
      .cast<py::list>();

  // Unfold dimensions
  linalg::TTEngine::Tensors x_cores;
  x_cores.reserve(num_cores);

  for (size_t i = 0; i < num_cores; i++) {
    const auto& a_core = a_cores[i];
    torch::Tensor x_core = py_x_cores[i].cast<torch::Tensor>();
    x_cores.push_back(x_core.reshape(
      {x_core.size(0), a_core.size(1), a_core.size(2), x_core.size(2)}));
  }

  return linalg::TTEngine(x_cores, false);
}

linalg::TTEngine Acquire::amen_mm(const linalg::TTEngine& a,
  const linalg::TTEngine& b, int nswp, std::optional<linalg::TTEngine> x0,
  double eps, int rmax, int kickrank, int kick2, bool verbose)
{
  // Get the Python versions of the TTs
  py::object py_a = ttnte2torchtt(a);
  py::object py_b = ttnte2torchtt(b);

  // Handle the initial guess
  py::object py_x0 = py::none();
  if (x0.has_value()) {
    py_x0 = ttnte2torchtt(*x0);
  }

  return torchtt2ttnte(
    PackageManager::instance().torchtt.attr("_amen").attr("amen_mm")(py_a, py_b,
      py::arg("nswp") = nswp, py::arg("X0") = py_x0, py::arg("eps") = eps,
      py::arg("rmax") = rmax, py::arg("kickrank") = kickrank,
      py::arg("kick2") = kick2, py::arg("verbose") = verbose));
}

} // namespace ttnte::python::torchtt
