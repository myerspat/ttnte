#include "ttnte/parallel/request.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_Request(py::module_& m)
{
  using namespace ttnte::parallel;

  py::class_<Request>(m, "Request")
    .def(py::init<>())                      // Default constructor
    .def(py::init<int>(), py::arg("f_req")) // Explicit constructor

    .def("wait", &Request::wait)
    .def("get", &Request::get)
    .def("update", &Request::update, py::arg("f_req"))

    // =================================================================
    // wait_all Trick:
    // We use a lambda that takes pointers so we can mutate the actual
    // Python objects. Standard pybind11 STL conversion creates a copy
    // of the list, which means the underlying MPI_Request handles in
    // Python wouldn't get set to null/updated properly after waiting.
    // =================================================================
    .def_static(
      "wait_all",
      [](std::vector<Request*>& py_requests) {
        // 1. Copy out to a C++ vector
        std::vector<Request> cpp_requests;
        cpp_requests.reserve(py_requests.size());
        for (auto* req : py_requests) {
          cpp_requests.push_back(*req);
        }

        // 2. Call your actual C++ wait_all function
        Request::wait_all(cpp_requests);

        // 3. Write the mutated state (e.g., MPI_REQUEST_NULL) back to Python
        for (size_t i = 0; i < py_requests.size(); ++i) {
          py_requests[i]->update(cpp_requests[i].get());
        }
      },
      py::arg("requests"));
}
