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
#include <string>
#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace ttnte::cross {

/// @brief The discrete transport problem as seen by the cross: index sizes,
/// angular data, the spatial blocks (patches) of a fiber and the oracle that
/// sweeps fibers.
struct CrossProblem {
  /// Sweeps a request of fibers into a (num_fibers, num_space) buffer, block
  /// by block, direction-major then group (see SweepDriver::sweep_fibers()),
  /// calling `done` exactly once for every output row (in ranges, possibly
  /// concurrently, as fibers finish).
  using SweepFn =
    std::function<void(const std::vector<sweep::FiberBlock>& blocks,
      double* out, const sweep::FiberDone& done)>;

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
  /// quadrant/octant); at least one, all in range.
  std::vector<int64_t> initial_directions;
  /// Offset of each patch in a fiber, and its control-point grid shape.
  std::vector<int64_t> patch_offsets;
  std::vector<std::vector<int64_t>> patch_shapes;
  /// Per patch, the lower band (n_P, kd + 1) of its mass matrix (L2
  /// weighting in space); empty for no spatial weighting.
  std::vector<torch::Tensor> mass_bands;
  /// (num_groups) lethargy widths ln(E_{g-1} / E_g), all positive, for the
  /// energy weighting 1 / sqrt(du_g); empty for uniform weighting.
  std::vector<double> lethargy_widths;
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

  /// @brief State after a half-sweep of a run: forward (interpolation
  /// ranks, bounded by the index sets) or backward (truncated ranks, then the
  /// hold-out check).
  struct SweepRecord {
    bool forward = false;
    std::vector<int64_t> ranks;
    /// Fibers swept so far in the run.
    int64_t fibers = 0;
    /// Error of the hold-out pool relative to its own norm (backward only).
    double holdout_error = 0.0;
    /// RMS norms of the (weighted) hold-out pool fibers and of every fiber
    /// swept so far, the pool included (backward only, diagnostics): the
    /// hold-out error is relative to the pool's norm; each truncation is
    /// relative to its own projected superblock's norm.
    double holdout_rms = 0.0;
    double swept_rms = 0.0;
    /// Estimated global relative error (backward only, see
    /// CrossConfig::global_holdout); NaN when the pool alone accepted.
    double global_error = 0.0;
    /// Pairs forced into the index sets so far in the run (backward only,
    /// after this check).
    int64_t forced = 0;
  };

  /// @brief One request to the sweep oracle: the phase of run() that made it
  /// (init, superblock, check), its new fibers, sweep blocks (a direction
  /// list sharing one group list) and distinct directions, and its seconds.
  struct SweepCall {
    std::string phase;
    int64_t fibers = 0;
    int64_t blocks = 0;
    int64_t directions = 0;
    double seconds = 0.0;
  };

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
  /// An index set: per element, its (mu, gamma, s, E) components, -1 where
  /// the set does not fix that mode. `defined` is false until the set is
  /// first chosen in a run.
  struct IndexSet {
    bool defined = true;
    std::vector<int64_t> comp;
    int64_t size() const noexcept
    {
      return static_cast<int64_t>(comp.size() / 4);
    }
  };

  /// The candidates of one selection grouped into blocks. Candidate row
  /// (a, j) is S_a K_j: S the side vectors (rows of B_k on the left, of
  /// C_{k+1} on the right), K_j the core's slice at value j. A block is one
  /// side group (side rows sharing their non-space components) by one core
  /// group (core values sharing theirs); block A nJ + J has key keys[A nJ +
  /// J].
  struct CandidateBlocks {
    struct Group {
      std::array<int64_t, 4> comp;
      std::vector<int64_t> members;
    };
    /// (ms, q) side vectors; (nk, q, r) core slices (a view of the core).
    torch::Tensor Sv, K;
    /// (ms, nk, r) candidate rows, side rows in group order (only when
    /// every side group is one row).
    torch::Tensor T;
    int64_t ms = 0, nk = 0, q = 0, r = 0;
    std::vector<Group> side, core;
    /// Members of each core group as an index tensor.
    std::vector<torch::Tensor> J_index;
    std::vector<int64_t> keys;
    /// Every side group one row; also every core group (a block is one row).
    bool single = true, single_rows = true;
  };
  /// The state of one half pass of run(), recorded for tests (see
  /// set_recording()): its cores, the bases it chose (B_ forward, C_
  /// backward) and every index set after it.
  struct PassRecord {
    bool forward = false;
    std::vector<torch::Tensor> cores, bases;
    std::vector<IndexSet> left, right;
  };

  // =================================================================
  // Private data
  CrossProblem problem_;
  CrossConfig config_;
  /// The layout as configured (x, y, z separate if asked), and the solver's
  /// view of it with the spatial indices merged into one mode S
  /// (Layout::merge_space()); the cross runs on the latter, with K_ cores.
  tensor::Layout user_layout_;
  tensor::Layout layout_;
  int64_t K_ = 0;
  /// Per core k of layout_: its modes in order (MU, GA, S, E), their sizes,
  /// and n_[k] their product (the core's mode size n_k).
  std::vector<std::vector<int>> core_modes_;
  std::vector<std::vector<int64_t>> core_dims_;
  std::vector<int64_t> n_;
  /// Size of each mode, indexed by Mode: N_mu, N_gamma, N_x (fiber length),
  /// G; and the number of directions N_mu N_gamma.
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

  /// Nested index sets: left_[k] = I_{<=k-1} (rows of the cores before k),
  /// right_[k] = I_{>k-1} (columns of the cores from k on); left_[0] and
  /// right_[K_] hold one empty element. Their bases and pseudo-inverses
  /// follow (undefined: identity).
  std::vector<IndexSet> left_;
  std::vector<IndexSet> right_;
  /// Right index sets of the last run (CrossConfig::warm_start).
  std::vector<IndexSet> prior_right_;
  /// Bases at the chosen rows, (rows, r): B_ of left sets, C_ of right sets;
  /// PL_[k] = B_[k]^+ and PR_[k] = (C_[k]^+)^T project a superblock's rows
  /// and columns onto the bond (d261026 l. 82).
  std::vector<torch::Tensor> B_;
  std::vector<torch::Tensor> PL_;
  std::vector<torch::Tensor> C_;
  std::vector<torch::Tensor> PR_;
  /// Cores (r_l, n_k, r_r), weighted while running and unweighted after.
  std::vector<torch::Tensor> cores_;

  /// Swept fibers (weighted): one block per sweep request, never moved, and
  /// each pair's fiber within it (null: not swept). Blocks come from
  /// torch's CPU allocator, which puts large ones on huge pages when
  /// THP_MEM_ALLOC_ENABLE=1.
  std::vector<c10::DataPtr> fiber_blocks_;
  std::vector<const double*> slot_;
  /// Pairs the fit has used (superblocks, init): never out of sample.
  std::vector<uint8_t> fitted_;
  int64_t num_swept_ = 0;
  /// Sum of squared norms of the swept (weighted) fibers, and the RMS norm
  /// of the hold-out pool at the last check.
  double swept_sq_ = 0.0;
  double holdout_rms_ = 0.0;
  /// Pool and estimated global errors of the last check.
  double pool_error_ = 0.0;
  double global_error_ = 0.0;
  /// Pairs forced into every index set (failed hold-out), residual-chosen
  /// block keys per set of cores (bit mask), and the hold-out pool.
  std::vector<int64_t> force_;
  std::map<uint32_t, std::vector<int64_t>> grow_;
  std::vector<int64_t> pool_;
  std::mt19937_64 rng_;
  /// Separate stream for the swept-fiber sample of the global estimate, so
  /// the hold-out pool does not depend on it.
  std::mt19937_64 sample_rng_;
  /// Record every half pass (testing aid), and the records of the last run.
  bool record_ = false;
  std::vector<PassRecord> passes_;
  /// Form every block Gram in factored form (testing aid).
  bool factored_grams_ = false;

  /// Results of the last run.
  double holdout_error_ = 0.0;
  bool converged_ = false;
  bool stalled_ = false;
  std::vector<SweepRecord> history_;
  int64_t sweeps_done_ = 0;
  int64_t total_fibers_ = 0;
  double sweep_seconds_ = 0.0;
  double seconds_ = 0.0;
  /// Seconds per phase of run(), accumulated over every run.
  std::map<std::string, double> timings_;
  /// Every sweep request, over every run.
  std::vector<SweepCall> sweep_calls_;
  /// Concatenated Gaussian sketches of the initial directions, by (number of
  /// initial directions, sketch rows): they depend on nothing else (the
  /// initial directions are fixed), so they are made once.
  std::map<std::pair<int64_t, int64_t>, torch::Tensor> sketches_;

  // =================================================================
  // Protected constructors
  LayoutCross(CrossProblem problem, CrossConfig config);

  // =================================================================
  // Private methods
  [[nodiscard]] static std::string ctx(const std::string& f)
  {
    return "ttnte::cross::LayoutCross::" + f;
  }
  /// @brief Clear everything of the previous run: fiber store, pool, forced
  /// and grown keys, index sets (left_[0] and right_[K_] empty, the others
  /// undefined), bases, projections and cores.
  void reset();
  /// @brief The key of a block: one integer for a choice of (mu, gamma, E),
  /// each -1 when the block leaves that mode free. A block is the set of
  /// (direction, group) fibers it covers, with all their spatial values, so
  /// the key ignores space; equal keys are the same sweeps.
  int64_t key(int64_t mu, int64_t ga, int64_t e) const noexcept;
  /// @brief key() of the (mu, gamma, s, E) components `comp` (space ignored).
  int64_t row_key(const int64_t* comp) const noexcept;
  /// @brief Write the components of value j of core k's mode (C order over
  /// the core's modes) into comp, leaving the other modes untouched.
  void decode(size_t k, int64_t j, int64_t* comp) const;
  /// @return The fiber index d * G + g of complete components (mu, gamma,
  /// E; gamma -1 in 1-D), the index of the fiber store.
  int64_t pair_of(const int64_t* comp) const;
  /// @brief Make every fiber of `pairs` (d * G + g) available: sweep those
  /// not yet cached in one oracle request (directions sharing a group list
  /// form one block), weight them as they finish (R_P, sqrt(w_d),
  /// 1 / sqrt(du_g), inside the sweep's tasks), store them in a new block of
  /// the fiber store, and record the request (SweepCall). Pairs requested by
  /// any phase but "check" are marked fitted_ (in sample from then on).
  /// @param phase The phase of run() making the request: init, superblock,
  /// or check.
  void ensure(std::vector<int64_t> pairs, const char* phase);
  /// @return Whether fiber `pair` is in the store.
  bool swept(int64_t pair) const { return slot_[pair] != nullptr; }
  /// @brief Core k's index as j = (outer * Ns + s) * inner + i, with s the
  /// space index: {outer, inner}, or {0, 0} if core k has no space mode.
  std::pair<int64_t, int64_t> space_split(size_t k) const;
  /// @brief Apply R_P (or R_P^{-1}) of every patch to `count` contiguous
  /// fibers at `rows`, serially (callers parallelize over rows).
  void apply_mass(double* rows, int64_t count, bool inverse) const;
  /// @return The stored (weighted) fiber of `pair`, num_space values; it
  /// must be swept.
  const double* fiber(int64_t pair) const { return slot_[pair]; }

  /// @brief Initial right index sets right_[K_ - 1] ... right_[1]:
  /// sweep every group of the initial directions D_0, choose the
  /// initial energies J_0 as the column
  /// pivots of a Gaussian sketch of those fibers (pivot_columns, the sketch
  /// doubled until the pivots settle), and keep, from k = K - 1 down, the
  /// candidates of core k x right_[k + 1] whose angle and energy components
  /// lie in D_0 and J_0 (four random blocks if none does).
  void init_right();
  /// @brief The index set of the chosen superblock rows of core k: left_[k]
  /// extended by core k (left), or core k by right_[k + 1] (right).
  IndexSet candidates(
    size_t k, bool left, const std::vector<int64_t>& rows) const;
  /// @return The block key of every candidate (j, l) of core k x
  /// right_[k + 1], at index j * |right_[k + 1]| + l.
  std::vector<int64_t> candidate_keys_right(size_t k) const;
  /// @return The block key of every element of `set`.
  std::vector<int64_t> set_keys(const IndexSet& set) const;
  /// @brief The superblock S_k = A[I_{<=k-1}, N_k, I_{>k}] of
  /// core k from the fiber store, sweeping the missing fibers
  /// first (ensure()). Whichever of the left rows, the core or the right
  /// columns holds the space index is enumerated over space while the copy
  /// reads whole fibers.
  /// @param raw Return S_k itself, (|left_[k]|, n_k, |right_[k + 1]|);
  /// otherwise the projected P_L S_k P_R, (r_{k-1}, n_k,
  /// r_k), applying first whichever projection leaves fewer multiplies.
  torch::Tensor superblock(size_t k, bool raw);
  /// @return The cost of a block (key()): its fibers not yet swept, the
  /// sweeps choosing it would add.
  double block_cost(int64_t key) const;
  /// @return The keys of the forced pairs (force_) restricted to the modes
  /// held by the cores in `mask` (a bit per core), for the index sets of
  /// those cores: each forced pair must stay in every set it projects to.
  std::vector<int64_t> forced_keys(uint32_t mask) const;
  /// @return The candidates of core k (left: left_[k] x the core's values;
  /// right: the core's values x right_[k + 1]) grouped into blocks, with
  /// their rows T when every side group is one row.
  /// @param core (q, n_k, r) on the left, (r, n_k, q) on the right.
  CandidateBlocks candidate_blocks(
    size_t k, bool left, const torch::Tensor& core) const;
  /// @return (nb, r, r) block Grams sum_{a in A, j in J} (S_a K_j)^T S_a K_j:
  /// from the rows T by batched products when every side group is one row,
  /// else (or if `factored`) as sum_{j in J} K_j^T H_A K_j, H_A = S_A^T S_A.
  torch::Tensor block_grams(const CandidateBlocks& cb, bool factored) const;
  /// @return Recorded half pass i (set_recording()), checking i.
  const PassRecord& recorded(int64_t i) const;
  /// @brief Choose the next index set of core k among its candidates
  /// (left: left_[k] x the core's values; right: the core's values x
  /// right_[k + 1]) by block selection on block Grams, computed in factored
  /// form when cheaper, and evaluate the new basis at the chosen rows only.
  /// @return The chosen candidate rows (ascending), the basis there, and
  /// its Gram matrix B^T B when selected by block Grams (undefined for
  /// single rows, whose B is small).
  std::tuple<std::vector<int64_t>, torch::Tensor, torch::Tensor>
  select_candidates(size_t k, bool left, const torch::Tensor& core,
    const std::vector<int64_t>& keep);
  /// @return The rank cap (CrossConfig::max_rank; none if <= 0).
  int64_t max_rank() const noexcept;
  /// @brief Truncated SVD at relative tolerance `rel_tol` and the rank cap,
  /// through the Gram matrix when config().gram_svd.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, double, double> svd(
    const torch::Tensor& A, double rel_tol) const;
  /// @brief Residual bond growth (Alg. d261021:grow, l. 326-330; K-core
  /// Alg. crossK line 38): if the last core's fit leaves a residual above
  /// tol * total, add the blocks (core values by key, space summed) with the
  /// largest residual energy that are neither in `current` nor already grown,
  /// the fewest covering growth_share of it but at most
  /// max(growth_max_blocks, half the blocks of `current`), to grow_[mask], so
  /// the next selection of those cores must keep them.
  /// @param col (n_k) residual energy per value of core k's mode.
  /// @param total Norm of the superblock the residual is relative to.
  /// @param mask The cores (bit per core) whose index sets take the keys.
  /// @param current The index set being grown (its block count).
  void grow_from_residual(const torch::Tensor& col, double total, size_t k,
    uint32_t mask, const IndexSet& current);
  /// @brief Forward pass (Alg. crossK lines 30-38): for k = 0 ... K - 2, the
  /// projected superblock's truncated SVD gives core k (share
  /// forward_truncation_share of the per-bond tolerance), selection chooses
  /// left_[k + 1] and its basis B_[k + 1], and PL_[k + 1] = B^+. The last
  /// core is fitted on its raw superblock, and its residual grows the bond
  /// (grow_from_residual()). The right sets are used as interpolation (PR_
  /// cleared), so ranks may grow up to their size.
  void forward();
  /// @brief Backward pass (Alg. crossK lines 39-47): for k = K - 1 ... 1, the
  /// projected superblock's truncated SVD (full per-bond share) gives core
  /// k, selection chooses right_[k] (keeping its previous keys) and its
  /// basis C_[k], and PR_[k] = (C^+)^T; core 0 is the left end fitted on its
  /// raw superblock.
  void backward();
  /// @brief Hold-out check: sweeps new pool fibers and sets pool_error_.
  /// A pool within holdout_factor * tol accepts (global_error_ NaN);
  /// otherwise, with global_holdout, the global estimate decides. The worst
  /// pool pairs are forced if the pass is not accepted.
  /// @return The error the pass was accepted or rejected on.
  double check();
  /// @brief Turn the weighted cores into the angular flux's (d261021
  /// Section "Moments from the cores"): divide the angle modes by sqrt(w_d)
  /// (split as sqrt(w_mu) sqrt(w_gamma) when they sit in different cores),
  /// multiply energy by sqrt(du_g), and solve R_P on each patch's block of
  /// the space mode (apply_mass(inverse)).
  void unweight();
  /// @brief moments() when every angle mode leads its core (joint polar and
  /// azimuthal at the front of one core, or each at the front of its own):
  /// the angle cores are contracted with all moments' weights in one product
  /// and the other cores chained with the moments as a batch, so no core is
  /// copied per moment.
  /// @param Yw (N_Omega, num_moments) quadrature-weighted harmonics.
  /// @return (num_moments, G, num_space), or undefined if the layout does
  /// not qualify.
  torch::Tensor moments_batched(const torch::Tensor& Yw) const;

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
  /// BATCH or WEIGHT, space ALL or LIST; energy any kind. LIST/BATCH data
  /// are int64 indices in range, WEIGHT data float64, one per index.
  /// @param angle_matrix (N_mu, N_gamma) joint angle weights, used when both
  /// angle modes are WEIGHT and share a core.
  /// @return (B, |S|, |E|): B the batch size (1 without a BATCH mode), |E| 1
  /// when energy is batched.
  torch::Tensor contract(const std::array<Selector, 4>& select,
    const torch::Tensor& angle_matrix = {}) const;

  /// @return (num_moments, G, num_space) flux moments.
  torch::Tensor moments() const;
  /// @return (|pairs|, num_space) fibers of the TT at (direction, group)
  /// pairs (d * G + g): each pair's left and right factors, and one product
  /// per slice of the space core, so no slice is copied per pair. For
  /// scattered pairs (the hold-out check); evaluate() contracts a grid of
  /// directions x groups with shared factors instead.
  torch::Tensor evaluate_pairs(const std::vector<int64_t>& pairs) const;
  /// @return (|directions|, |groups|, |space|) angular flux values.
  torch::Tensor evaluate(const std::vector<int64_t>& directions,
    const std::vector<int64_t>& groups,
    const std::vector<int64_t>& space) const;
  /// @return The TT of patch `p` in the configured layout (x/y/z split as
  /// requested), rounded at the tolerance: cores (r_l, n_k, r_r), n_k the
  /// product of the core's index sizes. The rounding acts on the unweighted
  /// cores, so this tolerance is Euclidean in the coefficients, not the
  /// cross's L2-weighted one.
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
  /// @return The error the last run's final check was accepted or rejected
  /// on: the pool's, or the global estimate when the pool failed and
  /// global_holdout is set.
  double holdout_error() const noexcept { return holdout_error_; }
  /// @return Estimated global relative error at the last check, NaN if the
  /// pool alone accepted it.
  double global_error() const noexcept { return global_error_; }
  /// @return One record per half-sweep of the last run.
  const std::vector<SweepRecord>& history() const noexcept { return history_; }
  bool converged() const noexcept { return converged_; }
  /// @return Whether the last run stopped on stalled passes (see
  /// CrossConfig::stall_passes).
  bool stalled() const noexcept { return stalled_; }
  int64_t sweeps_done() const noexcept { return sweeps_done_; }
  double sweep_seconds() const noexcept { return sweep_seconds_; }
  double seconds() const noexcept { return seconds_; }
  /// @return Seconds per phase of run(), accumulated over every run and
  /// disjoint except weight_cpu: sweep (the oracle, including the weighting
  /// of finished fibers that runs inside it), weight_cpu (seconds in the
  /// weighting callbacks summed over the threads running them, a CPU time
  /// overlapping sweep, not a phase of its own), gather (copying
  /// fibers into superblocks), project (B^+ and C^+ on superblocks), svd
  /// (truncations and end-core fits), select, basis (the left inverses of the
  /// new bases), init and check (both without their sweeps), unweight.
  const std::map<std::string, double>& timings() const noexcept
  {
    return timings_;
  }
  /// @return Every sweep request of every run (see SweepCall).
  const std::vector<SweepCall>& sweep_calls() const noexcept
  {
    return sweep_calls_;
  }
  const CrossConfig& config() const noexcept { return config_; }

  // =================================================================
  // Testing aids (read-only views of the cross's index bookkeeping)
  /// @return (|set|, 4) int64 components (mu, gamma, s, E; -1 free) of
  /// left_[k] (left) or right_[k] after the last run.
  torch::Tensor index_set(bool left, int64_t k) const;
  /// @return (n_k, 4) int64 components of every value of core k's mode
  /// (decode()).
  torch::Tensor core_components(int64_t k) const;
  /// @return key() of (mu, gamma, E), -1 for free modes.
  int64_t block_key(int64_t mu, int64_t ga, int64_t e) const noexcept
  {
    return key(mu, ga, e);
  }
  /// @return block_cost() of a key: its fibers not in the store (after a
  /// run the store is released, so every fiber of the block).
  double unswept_in_block(int64_t key) const { return block_cost(key); }
  /// @return The pairs forced into the index sets in the last run, in the
  /// order they were forced (SweepRecord::forced counts them per check).
  const std::vector<int64_t>& forced_pairs() const noexcept { return force_; }
  /// @return forced_keys() for the cores first ... last.
  std::vector<int64_t> forced_block_keys(int64_t first, int64_t last) const;
  /// @return grow_[mask] for the cores first ... last: the block keys the
  /// residual growth added in the last run.
  std::vector<int64_t> grown_block_keys(int64_t first, int64_t last) const;
  /// @return The block Grams of core k's candidates for a given core
  /// ((q, n_k, r) left, (r, n_k, q) right) against the current side basis
  /// (B_[k] or C_[k + 1]), by the path selection uses or all factored, and
  /// the blocks' keys.
  std::tuple<torch::Tensor, std::vector<int64_t>> candidate_grams(
    int64_t k, bool left, const torch::Tensor& core, bool factored) const;
  /// @brief Form every block Gram in factored form during selection.
  void set_factored_grams(bool on) noexcept { factored_grams_ = on; }
  /// @brief Record every half pass of the next runs (recorded_*()).
  void set_recording(bool on) noexcept { record_ = on; }
  /// @return Half passes recorded in the last run.
  int64_t num_recorded() const noexcept
  {
    return static_cast<int64_t>(passes_.size());
  }
  /// @return Whether recorded half pass i was a forward pass.
  bool recorded_forward(int64_t i) const;
  /// @return The cores after recorded half pass i (before unweighting).
  std::vector<torch::Tensor> recorded_cores(int64_t i) const;
  /// @return The bases after recorded half pass i: B_ (forward) or C_
  /// (backward), (|set|, rank) per bond, undefined where not chosen.
  std::vector<torch::Tensor> recorded_bases(int64_t i) const;
  /// @return index_set() after recorded half pass i.
  torch::Tensor recorded_index_set(int64_t i, bool left, int64_t k) const;
  void set_tolerance(double tol) { config_.tol = tol; }
  /// @brief Drop the cores of the last run (see
  /// CrossConfig::keep_angular_flux).
  void release() { cores_.clear(); }
  /// @return Whether the cores of the last run are held.
  bool has_tt() const noexcept
  {
    return static_cast<int64_t>(cores_.size()) == K_ && K_ > 0 &&
           cores_[0].defined();
  }
};

} // namespace ttnte::cross
