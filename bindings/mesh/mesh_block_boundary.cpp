#include "ttnte/mesh/mesh_block_boundary.hpp"
#include <sstream>
#include <torch/extension.h>

namespace py = pybind11;

void register_Boundary(py::module_& m)
{
  using namespace ttnte::mesh;

  // BoundaryMapping struct
  py::class_<BoundaryMapping>(m, "BoundaryMapping")
    .def(
      py::init([](std::vector<bool> flip_vec, std::vector<int64_t> perm_vec) {
        BoundaryMapping bm;
        bm.flip = c10::SmallVector<bool, 2>(flip_vec.begin(), flip_vec.end());
        bm.perm =
          c10::SmallVector<int64_t, 2>(perm_vec.begin(), perm_vec.end());
        return bm;
      }),
      py::arg("flip") = std::vector<bool> {},
      py::arg("perm") = std::vector<int64_t> {})
    .def("__repr__",
      [](const BoundaryMapping& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      })

    .def_property(
      "flip",
      [](const BoundaryMapping& self) {
        return std::vector<bool>(self.flip.begin(), self.flip.end());
      },
      [](BoundaryMapping& self, const std::vector<bool>& v) {
        self.flip = c10::SmallVector<bool, 2>(v.begin(), v.end());
      })
    .def_property(
      "perm",
      [](const BoundaryMapping& self) {
        return std::vector<int64_t>(self.perm.begin(), self.perm.end());
      },
      [](BoundaryMapping& self, const std::vector<int64_t>& v) {
        self.perm = c10::SmallVector<int64_t, 2>(v.begin(), v.end());
      });

  // NeighborInfo struct
  py::class_<NeighborInfo>(m, "NeighborInfo")
    .def(py::init([](int64_t global_id, size_t face_id, int mpi_rank,
                    const BoundaryMapping& mapping) {
      return NeighborInfo {global_id, face_id, mpi_rank, mapping};
    }),
      py::arg("global_id"), py::arg("face_id"), py::arg("mpi_rank"),
      py::arg("mapping") = BoundaryMapping())
    .def("__repr__", &NeighborInfo::to_string)
    .def("__str__",
      [](const NeighborInfo& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      })

    // The const members must be bound as readonly!
    .def_readwrite("gid", &NeighborInfo::gid)
    .def_readwrite("fid", &NeighborInfo::fid)
    .def_readwrite("mpi_rank", &NeighborInfo::mpi_rank)
    .def_readwrite("mapping", &NeighborInfo::mapping);

  // Boundary struct
  py::class_<BoundaryInfo>(m, "BoundaryInfo")
    .def(py::init<size_t, bool>(), py::arg("dim"), py::arg("is_upper"))
    .def("add_connection", &BoundaryInfo::add_connection, py::arg("ninfo"))
    .def("__repr__",
      [](const BoundaryInfo& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      })

    .def_property_readonly("fid", &BoundaryInfo::get_fid)
    .def_property_readonly("type", &BoundaryInfo::get_type)
    .def_property_readonly("connections", [](const BoundaryInfo& self) {
      const auto& conns = self.get_connections();
      return std::vector<NeighborInfo>(conns.begin(), conns.end());
    });
}
