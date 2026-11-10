#include "ttnte/sweep/patch_discretization.hpp"
#include "ttnte/sweep/extruded_solver.hpp"
#include "ttnte/sweep/sum_factorization.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_PatchDiscretization(py::module_& m)
{
  using namespace ttnte::sweep;

  py::class_<FaceGeometry>(m, "FaceGeometry")
    .def_readonly("dim", &FaceGeometry::dim)
    .def_readonly("is_upper", &FaceGeometry::is_upper)
    .def_readonly("shape", &FaceGeometry::shape)
    .def_readonly("dofs", &FaceGeometry::dofs)
    .def_readonly("normal", &FaceGeometry::normal)
    .def_readonly("weights", &FaceGeometry::weights)
    .def_readonly("area", &FaceGeometry::area)
    .def_readonly("pair_q", &FaceGeometry::pair_q,
      "Face pair list: quadrature point of each (point, test, trial) pair")
    .def_readonly("pair_a", &FaceGeometry::pair_a)
    .def_readonly("pair_b", &FaceGeometry::pair_b)
    .def_readonly("pair_val", &FaceGeometry::pair_val,
      "R_a(q) R_b(q), face-local dofs a, b")
    .def("upwind_weights", &FaceGeometry::upwind_weights, py::arg("omega"),
      py::arg("outflow"),
      "(D, Qf) clipped, weighted normal components (Omega . n)_+/- w.");

  py::class_<PatchDiscretization, PatchDiscretization::Ptr>(
    m, "PatchDiscretization")
    .def(py::init([](const ttnte::cad::Patch::Ptr& patch) {
      return PatchDiscretization::create(*patch);
    }),
      py::arg("patch"))
    .def("streaming_values", &PatchDiscretization::streaming_values,
      py::arg("omega"))
    .def("to_band", &PatchDiscretization::to_band, py::arg("values"))
    .def_property_readonly("band_order", &PatchDiscretization::band_order,
      "(N) dof at each band position (None: natural ordering).")
    .def(
      "evaluate_face",
      [](const PatchDiscretization::Ptr& self, int64_t dim, bool is_upper,
        const torch::Tensor& u) {
        auto fp = self->evaluate_face(dim, is_upper, u);
        return py::make_tuple(fp.R, fp.index, fp.x, fp.normal);
      },
      py::arg("dim"), py::arg("is_upper"), py::arg("u"),
      "Face basis values, face-local indices, points and scaled outward "
      "normals at (Q, ndim - 1) parametric points.")
    .def("to_dense", &PatchDiscretization::to_dense, py::arg("values"))
    .def("apply_mass", &PatchDiscretization::apply_mass, py::arg("x"))
    .def(
      "apply_matrix_free",
      [](const PatchDiscretization::Ptr& self, const torch::Tensor& omega,
        double sigma, const torch::Tensor& u) {
        auto op = ttnte::sweep::SumFactorization::create(*self);
        auto w = omega.to(torch::kFloat64).contiguous();
        auto x = u.to(torch::kFloat64).contiguous();
        auto y = torch::empty_like(x);
        ttnte::sweep::SumFactorization::Workspace work;
        op->apply(w.data_ptr<double>(), sigma, x.data_ptr<double>(),
          y.data_ptr<double>(), work);
        return y;
      },
      "y = L(omega, sigma) u applied matrix-free by sum factorization",
      py::arg("omega"), py::arg("sigma"), py::arg("u"))
    .def("local_operator", &PatchDiscretization::local_operator,
      py::arg("omega"), py::arg("sigma_t"),
      "Dense (N, N) local operator sigma_t M - Omega . G + sum_f B_f^+.")
    .def("face", &PatchDiscretization::face, py::arg("dim"),
      py::arg("is_upper"), py::return_value_policy::reference_internal)
    .def_property_readonly("ndim", &PatchDiscretization::ndim)
    .def_property_readonly("shape", &PatchDiscretization::shape)
    .def_property_readonly("num_dofs", &PatchDiscretization::num_dofs)
    .def_property_readonly("num_pairs", &PatchDiscretization::num_pairs)
    .def_property_readonly("kl", &PatchDiscretization::kl)
    .def_property_readonly("ku", &PatchDiscretization::ku)
    .def_property_readonly(
      "mass",
      [](const PatchDiscretization& d) {
        return d.to_dense(d.mass_values().unsqueeze(0))[0];
      },
      "Dense mass matrix.")
    .def_property_readonly(
      "streaming",
      [](const PatchDiscretization& d) { return d.to_dense(d.grad_values()); },
      "Dense (ndim, N, N) streaming matrices (d_i R_v, R_u).")
    .def_property_readonly(
      "basis_integral", &PatchDiscretization::basis_integral);
}

void register_ExtrudedSolver(py::module_& m)
{
  using namespace ttnte::sweep;

  py::class_<ExtrudedSolver, ExtrudedSolver::Ptr>(m, "ExtrudedSolver",
    "Exact solver of an extruded 3-D patch's local operator (partially "
    "separable: Schur along the extruded axis, one in-plane band solve per "
    "mode)")
    .def_static("try_create", &ExtrudedSolver::try_create, py::arg("disc"),
      py::arg("tol") = 1e-10,
      py::arg("max_eigvec_cond") = DEFAULT_MAX_EIGVEC_COND,
      "The solver of an extruded patch, or None if the patch is not extruded. "
      "max_eigvec_cond: largest condition number of the axial eigenvectors "
      "for the diagonal (conjugate-pair) path, else Schur; 0 forces Schur")
    .def(
      "solve",
      [](const ExtrudedSolver& self, const torch::Tensor& omega, double sigma,
        const torch::Tensor& b) {
        auto w = omega.to(torch::kFloat64).contiguous();
        auto x = b.to(torch::kFloat64).contiguous().clone();
        ExtrudedSolver::Workspace work;
        self.solve(w.data_ptr<double>(), sigma, x.data_ptr<double>(), work);
        return x;
      },
      py::arg("omega"), py::arg("sigma"), py::arg("b"),
      "x with L(omega, sigma) x = b (L as PatchDiscretization.local_operator)")
    .def_property_readonly(
      "axis", &ExtrudedSolver::axis, "The separable (extruded) parametric axis")
    .def_property_readonly("physical_axis", &ExtrudedSolver::physical_axis,
      "The physical axis it maps to")
    .def_property_readonly("plane_bandwidth", &ExtrudedSolver::plane_bandwidth,
      "(kl, ku) of the in-plane band systems");
}
