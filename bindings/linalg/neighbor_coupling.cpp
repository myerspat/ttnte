#include "ttnte/linalg/neighbor_coupling.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_NeighborCoupling(py::module_& m)
{
  using namespace ttnte::linalg;
  using namespace ttnte::mesh;

  py::class_<NeighborCoupling>(m, "NeighborCoupling",
    "Coupling from a neighboring patch at an INTERNAL boundary face")
    .def(py::init([](size_t fid, NeighborInfo connection, Operator boundary_op,
                    size_t dim, bool is_upper, State recv_buffer) {
      NeighborCoupling c;
      c.fid = fid;
      c.connection = std::move(connection);
      c.boundary_op = std::move(boundary_op);
      c.dim = dim;
      c.is_upper = is_upper;
      c.recv_buffer = std::move(recv_buffer);
      return c;
    }),
      py::arg("fid"), py::arg("connection"), py::arg("boundary_op"),
      py::arg("dim"), py::arg("is_upper"), py::arg("recv_buffer") = State())

    // =================================================================
    // Fields
    .def_readwrite("fid", &NeighborCoupling::fid)
    .def_readwrite("connection", &NeighborCoupling::connection)
    .def_readwrite("boundary_op", &NeighborCoupling::boundary_op)
    .def_readwrite("dim", &NeighborCoupling::dim)
    .def_readwrite("is_upper", &NeighborCoupling::is_upper)
    .def_readwrite("recv_buffer", &NeighborCoupling::recv_buffer)
    .def_readwrite("send_buffer", &NeighborCoupling::send_buffer)
    .def_readonly("current_op", &NeighborCoupling::current_op,
      "Operator reducing a boundary-narrowed angular-flux State to the "
      "outgoing partial current on this face ((Omega . n)_+ upwind mask "
      "times the angular quadrature weights). May be undefined if not built "
      "for this format.")
    .def_readonly("sq_diff", &NeighborCoupling::sq_diff,
      "0.5 * ||J_new - J_old||^2 of the outgoing partial current on this "
      "face from the last local solve (LocalSolver::postsolve). 0 before the "
      "first sweep.")
    .def_readonly("sq_prev", &NeighborCoupling::sq_prev,
      "0.5 * ||J_old||^2 of the outgoing partial current on this face from "
      "the last local solve (the denominator contribution of the Schwarz "
      "error). 0 before the first sweep.")

    // =================================================================
    // Methods
    .def("set_recv_buffer", &NeighborCoupling::set_recv_buffer, py::arg("face"),
      py::arg("apply_mapping") = true, py::arg("eps") = 0,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())
    .def(
      "set_send_buffer", &NeighborCoupling::set_send_buffer, py::arg("face"));
}
