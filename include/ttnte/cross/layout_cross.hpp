#pragma once

#include "ttnte/cross/cross_config.hpp"
#include "ttnte/sweep/fiber_block.hpp"
#include "ttnte/tensor/layout.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>

namespace ttnte::cross {

/// @brief The discrete transport problem as seen by the cross: index sizes,
/// angular data, the spatial blocks (patches) of a fiber and the oracle that
/// sweeps fibers.
struct CrossProblem {
  /// Sweeps a request of fibers into a (num_fibers, num_space) buffer, block
  /// by block, direction-major then group (see SweepDriver::sweep_fibers()).
  using SweepFn = std::function<void(
    const std::vector<sweep::FiberBlock>& blocks, double* out)>;

  int64_t num_dim = 0;
  int64_t num_polar = 0;
  /// 1 in 1-D. Direction d = polar * num_azimuthal + azimuthal.
  int64_t num_azimuthal = 1;
  int64_t num_groups = 0;
  /// Fiber length (dofs of every patch, concatenated).
  int64_t num_space = 0;
  /// (N_Omega) quadrature weights.
  torch::Tensor weights;
  /// (N_Omega, num_moments) harmonics; moments are sum_d w_d Y_m(d) psi_d.
  torch::Tensor harmonics;
  /// Directions swept over all groups to start the cross (one per
  /// quadrant/octant).
  std::vector<int64_t> initial_directions;
  /// Offset of each patch in a fiber, and its control-point grid shape.
  std::vector<int64_t> patch_offsets;
  std::vector<std::vector<int64_t>> patch_shapes;
  /// Per patch, the lower band (n_P, kd + 1) of its mass matrix (L2
  /// weighting in space); empty for no spatial weighting.
  std::vector<torch::Tensor> mass_bands;
  SweepFn sweep;
};

/// @brief Sweep-sampled TT-cross of one transport solve over a chosen layout
/// of (polar, azimuthal, space, energy).
///
/// Every sample is an exact upwind sweep of one (direction, group) pair,
/// which returns the whole spatial fiber, so space is never crossed and
/// every cost is a count of distinct pairs. Bonds are crossed with block
/// pivots (all spatial values of one choice of the other indices),
/// oversampled and fitted by least squares. One forward and backward sweep
/// of a rank-adaptive one-site cross is repeated until an accumulated
/// hold-out set of out-of-sample fibers fits within holdout_factor * tol.
/// The moments, traces and TTs are contracted from the (unweighted) cores;
/// the angular flux is never formed.
class LayoutCross {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<LayoutCross>;

  /// Modes of the sampled tensor.
  enum Mode : int { MU = 0, GA = 1, S = 2, E = 3 };

  /// How a mode is treated in a contraction of the TT.
  struct Selector {
    enum class Kind : uint8_t { ALL, LIST, BATCH, WEIGHT };
    Kind kind = Kind::ALL;
    /// LIST/BATCH: int64 indices; WEIGHT: float64 weights.
    torch::Tensor data;
  };

private:
  // =================================================================
  // Private types
  /// Multi-indices over consecutive cores: per row, one flat index per core
  /// and the decoded (mu, gamma, s, e) components (-1 where absent).
  struct IndexSet {
    int64_t width = 0;
    std::vector<int64_t> idx;
    std::vector<int64_t> comp;
    int64_t size() const noexcept
    {
      return static_cast<int64_t>(comp.size() / 4);
    }
  };

  // =================================================================
  // Private data
  CrossProblem problem_;
  CrossConfig config_;
  tensor::Layout user_layout_;
  tensor::Layout layout_;
  int64_t K_ = 0;
  std::vector<std::vector<int>> core_modes_;
  std::vector<std::vector<int64_t>> core_dims_;
  std::vector<int64_t> n_;
  std::array<int64_t, 4> size_ {};
  int64_t num_directions_ = 0;

  /// Weights of the samples: sqrt(w_d) per direction, 1 / sqrt(du_g) per
  /// group, and the Cholesky factors of the patch mass matrices.
  std::vector<double> angle_weight_;
  std::vector<double> energy_weight_;
  std::vector<torch::Tensor> mass_factors_;
  std::vector<int64_t> mass_kd_;
  /// The same factors as dense upper triangular R_P for patches small enough
  /// to apply them to batches of fibers with BLAS-3 (undefined otherwise).
  std::vector<torch::Tensor> mass_dense_;
  /// Separable quadrature weights w_d = w_mu w_gamma.
  torch::Tensor polar_weight_;
  torch::Tensor azimuthal_weight_;

  /// Index sets, their bases and pseudo-inverses (undefined: identity).
  std::vector<IndexSet> left_;
  std::vector<IndexSet> right_;
  std::vector<torch::Tensor> B_;
  std::vector<torch::Tensor> PL_;
  std::vector<torch::Tensor> C_;
  std::vector<torch::Tensor> PR_;
  /// Cores (r_l, n_k, r_r), weighted while running and unweighted after.
  std::vector<torch::Tensor> cores_;

  /// Swept fibers (weighted) by pair d * G + g.
  std::unordered_map<int64_t, int64_t> slot_;
  std::vector<double> fibers_;
  std::vector<uint8_t> swept_;
  int64_t num_swept_ = 0;
  /// Pairs forced into every index set (failed hold-out), residual-chosen
  /// block keys per set of cores (bit mask), and the hold-out pool.
  std::vector<int64_t> force_;
  std::map<uint32_t, std::vector<int64_t>> grow_;
  std::vector<int64_t> pool_;
  std::mt19937_64 rng_;

  /// Results of the last run.
  double holdout_error_ = 0.0;
  bool converged_ = false;
  int64_t sweeps_done_ = 0;
  int64_t total_fibers_ = 0;
  double sweep_seconds_ = 0.0;
  double seconds_ = 0.0;

  // =================================================================
  // Protected constructors
  LayoutCross(CrossProblem problem, CrossConfig config);

  // =================================================================
  // Private methods
  [[nodiscard]] static std::string ctx(const std::string& f)
  {
    return "ttnte::cross::LayoutCross::" + f;
  }
  void reset();
  int64_t key(int64_t mu, int64_t ga, int64_t e) const noexcept;
  int64_t row_key(const int64_t* comp) const noexcept;
  void decode(size_t k, int64_t j, int64_t* comp) const;
  int64_t pair_of(const int64_t* comp) const;
  void ensure(std::vector<int64_t> pairs);
  const double* fiber(int64_t pair) const
  {
    return &fibers_[slot_.at(pair) * size_[S]];
  }

  void init_right();
  IndexSet candidates_left(size_t k, const std::vector<int64_t>& rows) const;
  IndexSet candidates_right(size_t k, const std::vector<int64_t>& rows) const;
  std::vector<int64_t> candidate_keys_right(size_t k) const;
  std::vector<int64_t> set_keys(const IndexSet& set) const;
  torch::Tensor superblock(size_t k, bool raw);
  double block_cost(int64_t key) const;
  std::vector<int64_t> forced_keys(uint32_t mask) const;
  /// @brief Choose the next index set of core k among its candidates
  /// (left: left_[k] x the core's values; right: the core's values x
  /// right_[k + 1]) by block selection on block Grams, computed in factored
  /// form when cheaper, and evaluate the new basis at the chosen rows only.
  /// @return The chosen candidate rows (ascending) and the basis there.
  std::pair<std::vector<int64_t>, torch::Tensor> select_candidates(size_t k,
    bool left, const torch::Tensor& core, const std::vector<int64_t>& keep);
  int64_t max_rank() const noexcept;
  void grow_from_residual(const torch::Tensor& col, double total, size_t k,
    uint32_t mask, const IndexSet& current);
  void forward();
  void backward();
  double check();
  void unweight();

public:
  // =================================================================
  // Public methods
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new LayoutCross(std::forward<Args>(args)...));
  }

  /// @brief Run the cross for the current state of the oracle (one
  /// transport solve).
  void run();

  /// @brief Contract the TT over every mode with a selector.
  /// @param select One selector per mode (MU, GA, S, E). Angle modes must be
  /// BATCH or WEIGHT, space ALL or LIST, energy anything but WEIGHT.
  /// @param angle_matrix (N_mu, N_gamma) joint angle weights, used when both
  /// angle modes are WEIGHT and share a core.
  /// @return (B, |S|, |E|): B the batch size (1 without a BATCH mode), |E| 1
  /// when energy is batched.
  torch::Tensor contract(const std::array<Selector, 4>& select,
    const torch::Tensor& angle_matrix = {}) const;

  /// @return (num_moments, G, num_space) flux moments.
  torch::Tensor moments() const;
  /// @return (|directions|, |groups|, |space|) angular flux values.
  torch::Tensor evaluate(const std::vector<int64_t>& directions,
    const std::vector<int64_t>& groups,
    const std::vector<int64_t>& space) const;
  /// @return The TT of patch `p` in the configured layout (x/y/z split as
  /// requested), rounded at the tolerance: cores (r_l, n_k, r_r), n_k the
  /// product of the core's index sizes.
  std::vector<torch::Tensor> patch_tt(size_t p) const;

  // =================================================================
  // Public getters
  /// @return Cores (r_l, n_k, r_r) of the global TT in the solver layout
  /// (space merged), unweighted.
  const std::vector<torch::Tensor>& cores() const noexcept { return cores_; }
  const tensor::Layout& layout() const noexcept { return layout_; }
  const tensor::Layout& user_layout() const noexcept { return user_layout_; }
  std::vector<int64_t> ranks() const;
  int64_t num_fibers() const noexcept { return num_swept_; }
  int64_t total_fibers() const noexcept { return total_fibers_; }
  double holdout_error() const noexcept { return holdout_error_; }
  bool converged() const noexcept { return converged_; }
  int64_t sweeps_done() const noexcept { return sweeps_done_; }
  double sweep_seconds() const noexcept { return sweep_seconds_; }
  double seconds() const noexcept { return seconds_; }
  const CrossConfig& config() const noexcept { return config_; }
  void set_tolerance(double tol) { config_.tol = tol; }
  /// @brief Drop the cores of the last run (see CrossConfig::keep_tt).
  void release() { cores_.clear(); }
  /// @return Whether the cores of the last run are held.
  bool has_tt() const noexcept
  {
    return static_cast<int64_t>(cores_.size()) == K_ && K_ > 0 &&
           cores_[0].defined();
  }
};

} // namespace ttnte::cross
