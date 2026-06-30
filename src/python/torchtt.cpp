#include "ttnte/python/torchtt.hpp"
#include "ttnte/python/package_manager.hpp"
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace {

py::object ttnte2torchtt(
  const ttnte::linalg::TTEngine& engine, bool is_vec = false)
{
  py::list cores;

  for (const auto& core : engine.get_cores()) {
    cores.append(py::cast(is_vec ? core.squeeze(2) : core));
  }

  return ttnte::python::PackageManager::instance().torchtt.attr("TT")(cores);
}

ttnte::linalg::TTEngine torchtt2ttnte(
  const py::object& py_engine, bool is_vec = false)
{
  // Get the list of cores
  py::list py_cores = py_engine.attr("cores").cast<py::list>();

  ttnte::linalg::TTEngine::Tensors cores;
  for (auto core : py_cores) {
    cores.push_back(is_vec ? core.cast<torch::Tensor>().unsqueeze(2)
                           : core.cast<torch::Tensor>());
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
  std::optional<std::string> preconditioner) const
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
  double eps, int rmax, int kickrank, int kick2, bool verbose) const
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

linalg::TTEngine Acquire::function_interpolate(
  const std::function<torch::Tensor(const torch::Tensor&)>& func,
  const linalg::TTEngine& x, double eps,
  std::optional<linalg::TTEngine> start_tens, int nswp, int kick, int rmax,
  bool verbose) const
{
  py::cpp_function py_func = py::cpp_function(func);

  // Pass as a single object, NOT a list! This triggers eval_mv = False
  py::object py_x = ttnte2torchtt(x, true);

  py::object py_start = py::none();
  if (start_tens.has_value()) {
    py_start = ttnte2torchtt(*start_tens, true);
  }

  py::object py_result =
    PackageManager::instance()
      .torchtt.attr("interpolate")
      .attr("function_interpolate")(py_func, py_x, py::arg("eps") = eps,
        py::arg("start_tens") = py_start, py::arg("nswp") = nswp,
        py::arg("kick") = kick, py::arg("rmax") = rmax,
        py::arg("verbose") = verbose);

  return torchtt2ttnte(py_result, true);
}

linalg::TTEngine Acquire::function_interpolate(
  const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>& func,
  const std::vector<linalg::TTEngine>& xs, double eps,
  std::optional<linalg::TTEngine> start_tens, int nswp, int kick, int rmax,
  bool verbose) const
{
  // The Adapter: Bridge Python's [M, d] tensor to C++'s std::vector
  auto pybind_wrapper = [func](const torch::Tensor& pts) -> torch::Tensor {
    // pts is shape [M, d]. We slice it into 'd' separate 1D tensors.
    int64_t num_vars = pts.size(1);
    std::vector<torch::Tensor> unpacked_vars;
    unpacked_vars.reserve(num_vars);

    for (int64_t i = 0; i < num_vars; ++i) {
      unpacked_vars.push_back(pts.select(1, i));
    }

    // Call the user's generalized C++ function
    return func(unpacked_vars);
  };

  // Wrap our adapter, NOT the raw user function
  py::cpp_function py_func = py::cpp_function(pybind_wrapper);

  // Convert the vector of grid TTs into a Python list of torchtt.TT
  py::list py_xs;
  for (const auto& x : xs) {
    py_xs.append(ttnte2torchtt(x, true));
  }

  // Handle the initial guess (start_tens)
  py::object py_start = py::none();
  if (start_tens.has_value()) {
    py_start = ttnte2torchtt(*start_tens, true);
  }

  // Call torchtt.interpolate.function_interpolate
  py::object py_result =
    PackageManager::instance()
      .torchtt.attr("interpolate")
      .attr("function_interpolate")(py_func, py_xs, py::arg("eps") = eps,
        py::arg("start_tens") = py_start, py::arg("nswp") = nswp,
        py::arg("kick") = kick, py::arg("rmax") = rmax,
        py::arg("verbose") = verbose);

  // Convert back to C++ TTEngine
  return torchtt2ttnte(py_result, true);
}

linalg::TTEngine Acquire::dmrg_cross(
  const std::function<torch::Tensor(const torch::Tensor&)>& func,
  const std::vector<int64_t>& N, double eps, int nswp,
  std::optional<linalg::TTEngine> x0, int kick, int rmax, bool verbose,
  const torch::Device& device, const torch::ScalarType& dtype) const
{
  // Wrap the C++ lambda so Python can call it
  py::cpp_function py_func = py::cpp_function(func);

  // Convert the C++ shape vector to a Python list
  py::list py_N;
  for (int64_t n : N) {
    py_N.append(n);
  }

  // Handle the initial guess
  py::object py_start = py::none();
  if (x0.has_value()) {
    py_start = ttnte2torchtt(*x0, true);
  }

  // Call torchtt.interpolate.dmrg_cross
  // Note: torchTT uses slightly different kwargs across its modules.
  // dmrg_cross typically uses 'X0' and 'kickrank' (unlike
  // function_interpolate).
  py::object py_result =
    PackageManager::instance()
      .torchtt.attr("interpolate")
      .attr("dmrg_cross")(py_func, py_N, py::arg("eps") = eps,
        py::arg("nswp") = nswp, py::arg("x_start") = py_start,
        py::arg("kick") = kick, py::arg("dtype") = dtype,
        py::arg("device") = device, py::arg("rmax") = rmax,
        py::arg("verbose") = verbose);

  // Convert the resulting Python TT object back to a C++ TTEngine
  return torchtt2ttnte(py_result, true);
}

} // namespace ttnte::python::torchtt
