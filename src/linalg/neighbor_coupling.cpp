#include "ttnte/linalg/neighbor_coupling.hpp"

namespace ttnte::linalg {

void NeighborCoupling::set_recv_buffer(
  State face, bool apply_mapping, double eps, int64_t max_rank)
{
  if (apply_mapping) {
    const auto& mapping = connection.mapping;
    const size_t num_dim = mapping.flip.size() + 1;
    // num_angular inferred from total physical dims: angular + spatial + energy
    const size_t num_angular =
      static_cast<size_t>(face.ndimension()) - num_dim - 1;

    // Build the full physical-dimension permutation: angular and energy are
    // identity; spatial dims are reordered by mapping.perm so the face arrives
    // in this patch's coordinate ordering before the boundary operator is
    // applied.  State::permute_ dispatches to the right engine (TT or Dense).
    c10::SmallVector<int64_t, 6> perm;
    for (size_t k = 0; k < num_angular; ++k)
      perm.push_back(static_cast<int64_t>(k));
    for (size_t k = 0; k < num_dim; ++k)
      perm.push_back(static_cast<int64_t>(num_angular) + mapping.perm[k]);
    perm.push_back(static_cast<int64_t>(num_angular + num_dim));

    bool needs_perm = false;
    for (size_t k = 0; k < perm.size(); ++k) {
      if (perm[k] != static_cast<int64_t>(k)) {
        needs_perm = true;
        break;
      }
    }
    if (needs_perm)
      face.permute_(perm, eps, max_rank);

    // After the permutation the spatial dims are in target ordering, so dim d
    // correctly indexes the core/axis for target spatial dimension d.
    c10::SmallVector<int64_t, 4> flip_dims;
    size_t perp_idx = 0;
    for (size_t d = 0; d < num_dim; ++d) {
      if (d == dim)
        continue;
      if (mapping.flip[perp_idx])
        flip_dims.push_back(static_cast<int64_t>(num_angular + d));
      ++perp_idx;
    }
    if (!flip_dims.empty())
      face.flip_(flip_dims);
  }
  recv_buffer = std::move(face);
}

} // namespace ttnte::linalg
