#include "ttnte/linalg/tt_engine.hpp"
#include "ttnte/linalg/matrix_ops.hpp"
#include "ttnte/linalg/tt_ops.hpp"
#include "ttnte/utils/exception.hpp"
#include <cmath>

namespace ttnte::linalg {

// =================================================================
// Public constructors
TTEngine::TTEngine(const Tensors& cores, bool check_cores) : cores_(cores)
{
  if (check_cores) {
    if (cores.empty()) {
      throw utils::runtime_error(
        "ttnte::linalg::TTEngine::TTEngine", "`cores` is empty");
    }

    // Check the cores are all on the same device with the same data type
    const auto& device = cores_[0].device();
    const auto& dtype = cores_[0].dtype();

    // Iterate through each core and make sure they're the right shape
    int64_t left_rank = 1;
    for (auto& core : cores_) {
      if (core.device() != device || core.dtype() != dtype) {
        throw utils::runtime_error("ttnte::linalg::TTEngine::TTEngine",
          "All TT-cores must be on the same device with the same data type");
      }

      if (core.ndimension() != 3 && core.ndimension() != 4) {
        throw utils::runtime_error("ttnte::linalg::TTEngine::TTEngine",
          "All TT-cores must either be 3-dimensional for vectors or\n"
          "4-dimensional for matrices");
      }

      if (core.size(0) != left_rank) {
        throw utils::runtime_error("ttnte::linalg::TTEngine::TTEngine",
          "The right rank of core k must equal the left rank of core k + 1");
      }

      if (core.ndimension() == 3) {
        core = core.reshape({core.size(0), core.size(1), 1, core.size(2)});
      }
      core = core.contiguous();

      left_rank = core.size(3);
    }

    if (left_rank != 1) {
      throw utils::runtime_error("ttnte::linalg::TTEngine::TTEngine",
        "The last rank of the last core must equal to 1");
    }

  } else {
    cores_ = cores;
  }
}

// =================================================================
// Public methods
TTEngine TTEngine::clone_from(const Tensors& cores, bool check_cores)
{
  Tensors new_cores;
  new_cores.reserve(cores.size());

  for (const auto& core : cores) {
    new_cores.push_back(core.clone());
  }
  return TTEngine(std::move(new_cores), check_cores);
}

TTEngine TTEngine::to(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  Tensors new_cores;
  new_cores.reserve(cores_.size());

  for (const auto& core : cores_) {
    new_cores.push_back(
      core.to(device, dtype, non_blocking, copy, memory_format));
  }

  return TTEngine(std::move(new_cores));
}

TTEngine TTEngine::to(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return to(cores_[0].device(), dtype, non_blocking, copy, memory_format);
}
TTEngine TTEngine::to(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return to(device, cores_[0].scalar_type(), non_blocking, copy, memory_format);
}

TTEngine& TTEngine::to_(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  for (auto& core : cores_) {
    core = core.to(device, dtype, non_blocking, copy, memory_format);
  }
  return *this;
}

TTEngine& TTEngine::to_(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  return to_(cores_[0].device(), dtype, non_blocking, copy, memory_format);
}
TTEngine& TTEngine::to_(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format)
{
  return to_(
    device, cores_[0].scalar_type(), non_blocking, copy, memory_format);
}

TTEngine& TTEngine::to_(const torch::TensorOptions& options)
{
  for (auto& core : cores_) {
    core = core.to(options);
  }
  return *this;
}

double TTEngine::sum() const
{
  if (cores_.empty())
    return 0.0;
  if (cores_.size() == 1)
    return cores_[0].sum().item<double>();

  torch::Tensor v = torch::ones(
    {1, 1}, torch::TensorOptions().device(get_device()).dtype(get_dtype()));

  for (const auto& core : cores_) {
    torch::Tensor m = core.sum({1, 2});
    v = torch::mm(v, std::move(m));
  }

  return v.reshape({}).item<double>();
}

double TTEngine::norm() const
{
  if (cores_.empty())
    return 0.0;
  if (cores_.size() == 1)
    return cores_[0].norm(2).item<double>();

  // TODO: Take advantage of orthogonalization if it has already been applied
  torch::Tensor v = torch::ones(
    {1, 1}, torch::TensorOptions().device(get_device()).dtype(get_dtype()));

  for (const auto& core : cores_) {
    torch::Tensor w = torch::tensordot(std::move(v), core, {0}, {0});
    v = torch::tensordot(std::move(w), core, {0, 1, 2}, {0, 1, 2});
  }

  double result = v.reshape({}).item<double>();
  return (result > 0.0) ? std::sqrt(result) : 0.0;
}

TTEngine::Tensors TTEngine::tt_svd(
  torch::Tensor tensor, double eps, int64_t max_rank)
{
  // Case where there is only one dimension
  if (tensor.ndimension() == 1) {
    return Tensors {tensor.clone().reshape({1, tensor.size(0), 1})};
  }

  int64_t ndim = tensor.ndimension();
  double delta = tensor.norm(2).item<double>() * eps /
                 std::sqrt(static_cast<double>(ndim - 1));
  auto shape = tensor.sizes().vec();

  // Array of TT-cores
  Tensors cores;
  cores.reserve(ndim);

  int64_t r = 1;
  for (int64_t i = 0; i < ndim - 1; i++) {
    int64_t n = shape[i];

    // Perform truncated SVD
    auto [u, s, vh, _1, _2] =
      truncated_svd(tensor.reshape({r * n, -1}), delta, max_rank, false);

    // Get the right rank
    r = u.size(1);

    // Store the current core
    cores.push_back(u.reshape({-1, n, r}));

    // Contract the singular values back into the tensor
    tensor = s.unsqueeze(1) * vh;
  }

  // Push the last core into the array of tensors
  cores.push_back(tensor.reshape({r, shape.back(), 1}));
  return cores;
}

TTEngine& TTEngine::lr_orthogonalize_()
{
  // Iterate from left to right and perform QR
  for (size_t i = 0; i < cores_.size() - 1; i++) {
    torch::Tensor& core = cores_[i];
    auto shape = core.sizes().vec();

    // Perform QR decomposition
    auto [q, r] = torch::linalg_qr(core.reshape({-1, shape.back()}), "reduced");

    // Place Q back into the array of tensors
    shape.back() = q.size(1);
    core = q.reshape(shape);

    // Contract R with the next core
    cores_[i + 1] = torch::tensordot(r, cores_[i + 1], {1}, {0});
  }

  return *this;
}

TTEngine TTEngine::lr_orthogonalize() const
{
  auto copy = *this;
  copy.lr_orthogonalize_();
  return copy;
}

TTEngine& TTEngine::round_(double eps, int64_t max_rank)
{
  if (cores_.size() <= 1) {
    return *this;
  }
  assert(max_rank > 0);
  double delta = eps / std::sqrt(static_cast<double>(cores_.size() - 1));

  // Compute left-to-right orthogonalization
  lr_orthogonalize_();

  for (size_t i = cores_.size() - 1; i > 0; i--) {
    auto& rcore = cores_[i];
    auto& lcore = cores_[i - 1];

    // Get the shape of the right core
    auto shape = rcore.sizes().vec();

    // Perform truncated SVD
    bool is_first_step = i == cores_.size() - 1;
    auto [u, s, vh, new_delta, _1] = truncated_svd(
      rcore.reshape({shape.front(), -1}), delta, max_rank, is_first_step);
    if (is_first_step) {
      delta = new_delta;
    }

    // Update the rank and emplace the new core
    shape[0] = s.size(0);
    rcore = vh.reshape(shape);

    // Contract the left three tensors
    lcore = torch::tensordot(lcore, u * s, {-1}, {0});
  }

  return *this;
}

TTEngine TTEngine::round(double eps, int64_t max_rank) const
{
  auto copy = *this;
  copy.round_(eps, max_rank);
  return copy;
}

TTEngine TTEngine::from_dense(
  const torch::Tensor& tensor, double eps, int64_t max_rank)
{
  // Run the TT-SVD algorithm
  auto cores = tt_svd(tensor, eps, max_rank);

  // Reshape the cores into a vector
  for (auto& core : cores) {
    core.unsqueeze_(2);
  }

  return TTEngine(std::move(cores), false);
}

TTEngine TTEngine::from_dense(torch::Tensor tensor,
  const c10::SmallVector<int64_t, 6>& m_modes,
  const c10::SmallVector<int64_t, 6>& n_modes, double eps, int64_t max_rank,
  bool is_interleaved)
{
  if (m_modes.size() != n_modes.size()) {
    throw utils::runtime_error("ttnte::linalg::TTEngine::from_dense",
      "`m_modes` and `n_modes` must have the same length");
  }

  // Number of cores
  int64_t num_cores = n_modes.size();

  // Check if we need to permute the dimensions
  if (!is_interleaved) {
    // Full shape from all modes
    c10::SmallVector<int64_t, 12> full_shape;
    full_shape.insert(full_shape.end(), m_modes.begin(), m_modes.end());
    full_shape.insert(full_shape.end(), n_modes.begin(), n_modes.end());
    tensor = tensor.reshape(full_shape);

    // Permute to be interleaved
    c10::SmallVector<int64_t, 12> permutation;
    for (int64_t i = 0; i < num_cores; ++i) {
      permutation.push_back(i);
      permutation.push_back(i + num_cores);
    }
    tensor = tensor.permute(permutation);
  }

  // Combine dimensions for decomposition
  c10::SmallVector<int64_t, 6> decomp_shape;
  decomp_shape.reserve(num_cores);

  // Check the sizes provided match the tensor given
  {
    int64_t size = 1;
    for (size_t i = 0; i < num_cores; i++) {
      decomp_shape.push_back(n_modes[i] * m_modes[i]);
      size *= decomp_shape.back();
    }

    if (size != tensor.numel()) {
      throw utils::runtime_error("ttnte::linalg::TTEngine::from_dense",
        "The total size of `n_modes` and `m_modes` does not match\n"
        "`tensor.numel()`");
    }
  }

  // Reshape the given tensor and pass it to TT-SVD
  auto cores = tt_svd(tensor.reshape(decomp_shape), eps, max_rank);
  assert(cores.size() == num_cores);

  // Unravel the individual dimensions
  for (size_t i = 0; i < num_cores; i++) {
    auto& core = cores[i];
    assert(core.ndimension() == 3);
    core = core.reshape({core.size(0), m_modes[i], n_modes[i], core.size(-1)});
  }

  return TTEngine(std::move(cores), false);
}

torch::Tensor TTEngine::to_dense(bool interleave) const
{
  if (cores_.size() == 0) {
    return torch::Tensor();
  } else if (cores_.size() == 1) {
    return cores_[0].squeeze({0, -1});
  }

  int64_t num_cores = cores_.size();
  c10::SmallVector<int64_t, 12> dims(2 * num_cores, 0);
  dims[num_cores] = 1;
  dims[1] = 2;
  dims[num_cores + 1] = 3;

  // Contract all tensors
  auto result = torch::tensordot(cores_[0], cores_[1], {-1}, {0});
  for (size_t i = 2; i < cores_.size(); i++) {
    dims[i] = 2 * i;
    dims[i + num_cores] = dims[i] + 1;

    result = torch::tensordot(result, cores_[i], {-1}, {0});
  }

  if (interleave) {
    return result.squeeze({0, -1});
  } else {
    return result.squeeze({0, -1}).permute(dims);
  }
}

void TTEngine::from_buffer(const torch::Tensor& buffer)
{
  assert(buffer.numel() == get_numel() && buffer.ndimension() == 1);

  int64_t idx = 0;
  for (auto& core : cores_) {
    const auto& shape = core.sizes();
    core = buffer.narrow(0, idx, core.numel()).view(shape);
    idx += core.numel();
  }
}

TTEngine& TTEngine::neg_()
{
  cores_[0] = cores_[0].neg();
  return *this;
}

TTEngine TTEngine::zeros(const c10::SmallVector<int64_t, 6>& m_modes,
  std::optional<torch::Device> device, std::optional<torch::ScalarType> dtype)
{
  return zeros(
    m_modes, c10::SmallVector<int64_t, 6>(m_modes.size(), 1), device, dtype);
}

TTEngine TTEngine::zeros(const c10::SmallVector<int64_t, 6>& m_modes,
  const c10::SmallVector<int64_t, 6>& n_modes,
  std::optional<torch::Device> device, std::optional<torch::ScalarType> dtype)
{
  if (m_modes.size() != n_modes.size()) {
    throw utils::runtime_error("ttnte::linalg::tt::zeros",
      "The length of `m_modes` must equal the length of `n_modes`");
  }
  size_t num_cores = m_modes.size();
  auto options = at::TensorOptions().device(device).dtype(dtype);

  TTEngine::Tensors cores;
  cores.reserve(num_cores);

  for (size_t i = 0; i < num_cores; i++) {
    cores.push_back(torch::zeros({1, m_modes[i], n_modes[i], 1}, options));
  }

  return TTEngine(std::move(cores), false);
}

TTEngine TTEngine::ones(const c10::SmallVector<int64_t, 6>& m_modes,
  std::optional<torch::Device> device, std::optional<torch::ScalarType> dtype)
{
  return ones(
    m_modes, c10::SmallVector<int64_t, 6>(m_modes.size(), 1), device, dtype);
}

TTEngine TTEngine::ones(const c10::SmallVector<int64_t, 6>& m_modes,
  const c10::SmallVector<int64_t, 6>& n_modes,
  std::optional<torch::Device> device, std::optional<torch::ScalarType> dtype)
{
  if (m_modes.size() != n_modes.size()) {
    throw utils::runtime_error("ttnte::linalg::tt::ones",
      "The length of `m_modes` must equal the length of `n_modes`");
  }
  size_t num_cores = m_modes.size();
  auto options = at::TensorOptions().device(device).dtype(dtype);

  TTEngine::Tensors cores;
  cores.reserve(num_cores);

  for (size_t i = 0; i < num_cores; i++) {
    cores.push_back(torch::ones({1, m_modes[i], n_modes[i], 1}, options));
  }

  return TTEngine(std::move(cores), false);
}

TTEngine& TTEngine::transpose_(const c10::SmallVector<int64_t, 6>& core_idxs)
{
  if (core_idxs.empty()) {
    for (auto& core : cores_) {
      core = core.transpose(1, 2);
    }

  } else {
    for (const auto& idx : core_idxs) {
      assert(idx < cores_.size());
      cores_[idx] = cores_[idx].transpose(1, 2);
    }
  }

  return *this;
}

TTEngine TTEngine::transpose(
  const c10::SmallVector<int64_t, 6>& core_idxs) const
{
  auto result = *this;
  result.transpose_(core_idxs);
  return result;
}

TTEngine& TTEngine::diagonalize_(const c10::SmallVector<int64_t, 6>& core_idxs)
{
  // If the provided vector is empty, diagonalize all cores.
  c10::SmallVector<int64_t, 6> target_idxs = core_idxs;
  if (target_idxs.empty()) {
    for (int64_t i = 0; i < static_cast<int64_t>(cores_.size()); ++i) {
      target_idxs.push_back(i);
    }
  }

  for (int64_t idx : target_idxs) {
    TORCH_CHECK(idx >= 0 && idx < static_cast<int64_t>(cores_.size()),
      "ttnte::linalg::TTEngine::diagonalize_: Core index out of bounds");

    torch::Tensor& core = cores_[idx];
    auto shape = core.sizes();
    TORCH_CHECK(shape.size() == 4,
      "ttnte::linalg::TTEngine::diagonalize_: Expected 4D core (rl, m, n, rr)");

    int64_t rl = shape[0];
    int64_t m = shape[1];
    int64_t n = shape[2];
    int64_t rr = shape[3];

    // Ensure this core is actually a vector (one of the spatial dims is 1)
    TORCH_CHECK(m == 1 || n == 1,
      "ttnte::linalg::TTEngine::diagonalize_: Core must have at least one "
      "spatial mode of size 1 to be diagonalized");

    int64_t I = std::max(m, n);

    // Allocate the new square core (rl, I, I, rr) with zeros
    torch::Tensor new_core = torch::zeros({rl, I, I, rr}, core.options());

    // Squeeze out the singleton spatial dimension to get a 3D tensor: (rl, I,
    // rr)
    torch::Tensor vec_core = (m == 1) ? core.squeeze(1) : core.squeeze(2);

    // new_core.diagonal(0, 1, 2) extracts the diagonal across the spatial
    // modes. It returns a view of shape (rl, rr, I). We permute vec_core from
    // (rl, I, rr) to (rl, rr, I) to match the view and copy the data.
    new_core.diagonal(0, 1, 2).copy_(vec_core.permute({0, 2, 1}));

    // Replace the original core
    cores_[idx] = new_core;
  }

  return *this;
}

TTEngine TTEngine::diagonalize(
  const c10::SmallVector<int64_t, 6>& core_idxs) const
{
  auto copy = *this;
  copy.diagonalize_(core_idxs);
  return copy;
}

bool TTEngine::is_rank_one() const
{
  for (size_t i = 0; i < cores_.size() - 1; i++) {
    if (cores_[i].size(-1) != 1) {
      return false;
    }
  }
  return true;
}

TTEngine& TTEngine::expand_(const c10::SmallVector<int64_t, 6>& m_modes,
  const c10::SmallVector<int64_t, 6>& n_modes)
{
  for (size_t i = 0; i < cores_.size(); i++) {
    auto& core = cores_[i];
    auto shape = core.sizes().vec();

    if ((shape[1] != m_modes[i] && shape[1] != 1) ||
        (shape[2] != n_modes[i] && shape[2] != 1)) {
      throw utils::runtime_error("ttnte::linalg::TTEngine::expand_",
        "Cannot expand a dimension with size greater than 1");
    }

    shape[1] = m_modes[i];
    shape[2] = n_modes[i];
    core = core.expand(shape);
  }

  return *this;
}

TTEngine TTEngine::expand(const c10::SmallVector<int64_t, 6>& m_modes,
  const c10::SmallVector<int64_t, 6>& n_modes) const
{
  auto result = *this;
  result.expand_(m_modes, n_modes);
  return result;
}

TTEngine& TTEngine::contract_rank_dim_(size_t dim)
{
  if (cores_.size() <= 1 || dim >= cores_.size() - 1) {
    return *this;
  }

  // Cores involved
  auto& core_l = cores_[dim];
  const auto& core_r = cores_[dim + 1];

  // Dimension sizes in each core involved
  int64_t r_l = core_l.size(0);
  int64_t m_l = core_l.size(1);
  int64_t n_l = core_l.size(2);
  int64_t m_r = core_r.size(1);
  int64_t n_r = core_r.size(2);
  int64_t r_r = core_r.size(3);

  // Compute the contraction, permute the dimensions, and reshape
  cores_[dim] = torch::tensordot(cores_[dim], cores_[dim + 1], {-1}, {0})
                  .permute({0, 1, 3, 2, 4, 5})
                  .reshape({r_l, m_l * m_r, n_l * n_r, r_r});
  cores_.erase(cores_.begin() + dim + 1);
  return *this;
}

TTEngine TTEngine::contract_rank_dim(size_t dim) const
{
  auto result = *this;
  result.contract_rank_dim_(dim);
  return result;
}

c10::SmallVector<linalg::TTEngine, 6> TTEngine::meshgrid(const Tensors& vecs)
{
  size_t ndim = vecs.size();
  if (ndim == 0) {
    return {TTEngine()};
  }
  const auto& device = vecs[0].device();
  const auto& dtype = vecs[0].scalar_type();
  const auto& options = torch::TensorOptions().device(device).dtype(dtype);

  Tensors ones;
  ones.reserve(ndim);
  for (const auto& vec : vecs) {
    if (vec.ndimension() != 1 || vec.device() != device ||
        vec.scalar_type() != dtype) {
      throw utils::runtime_error("ttnte::linalg::TTEngine::meshgrid",
        "All tensors given must be 1-D, on the same device, and have the same\n"
        "data type");
    }
    ones.push_back(torch::ones({1, vec.size(0), 1, 1}, options));
  }

  c10::SmallVector<linalg::TTEngine, 6> grids;
  grids.reserve(ndim);
  for (size_t i = 0; i < ndim; i++) {
    Tensors grid = ones;
    grid[i] = vecs[i].reshape({1, -1, 1, 1});
    grids.emplace_back(grid, false);
  }

  return grids;
}

TTEngine& TTEngine::kron_(const TTEngine& other)
{
  cores_.append(other.cores_);
  return *this;
}

TTEngine& TTEngine::kron_(const torch::Tensor& other)
{
  if (other.ndimension() == 1) {
    cores_.push_back(other.reshape({1, -1, 1, 1}));
  } else if (other.ndimension() == 2) {
    cores_.push_back(other.reshape({1, other.size(0), other.size(1), 1}));
  } else if (other.ndimension() == 4) {
    if (other.size(0) != 1 || other.size(-1) != 1) {
      throw utils::runtime_error("ttnte::linalg::TTEngine::kron_",
        "The given tensor is 4-D with non-unit rank dimensions");
    }
    cores_.push_back(other);
  } else {
    throw utils::runtime_error("ttnte::linalg::TTEngine::kron_",
      "The given tensor was not 1-, 2-, or 4-D");
  }

  return *this;
}

TTEngine TTEngine::kron(const TTEngine& other) const
{
  auto copy = *this;
  copy.kron_(other);
  return copy;
}

TTEngine TTEngine::kron(const torch::Tensor& other) const
{
  auto copy = *this;
  copy.kron_(other);
  return copy;
}

torch::Tensor TTEngine::evaluate_at(const torch::Tensor& indices) const
{
  TORCH_CHECK(indices.dim() == 2,
    "Indices must be a 2D tensor of shape [BatchSize, NumCores].");
  TORCH_CHECK(indices.size(1) == static_cast<int64_t>(cores_.size()),
    "Indices width must match the number of TT cores.");

  bool is_vector = true;
  for (const auto& core : cores_) {
    if (core.size(2) != 1) {
      is_vector = false;
      break;
    }
  }
  TORCH_CHECK(is_vector, "The TT must be a TT-vector");

  int64_t batch_size = indices.size(0);
  int64_t num_cores = cores_.size();

  // Ensure indices are long integers and sit on the same device as the TT cores
  auto idx = indices.to(get_device(), torch::kInt64);

  // Extract matching physical coordinates for the entire batch out of Core 0
  auto tmp =
    cores_[0]
      .index_select(
        1, idx.select(1, 0)) // Shape: [Rank_Left(1), Batch, Mode_N, Rank_Right]
      .select(
        2, 0) // Shape: [Rank_Left(1), Batch, Rank_Right] (Assumes Mode_N=1)
      .permute({1, 0, 2}) // Shape: [Batch, Rank_Left(1), Rank_Right]
      .contiguous();

  // Iterate through the remaining cores using batched matrix multiplication
  for (int64_t k = 1; k < num_cores; ++k) {
    // Slice out the sub-matrices for this core across the entire batch
    auto sliced_core =
      cores_[k]
        .index_select(
          1, idx.select(1, k)) // Shape: [Rank_Left, Batch, Mode_N, Rank_Right]
        .select(2, 0)          // Shape: [Rank_Left, Batch, Rank_Right]
        .permute({1, 0, 2})    // Shape: [Batch, Rank_Left, Rank_Right]
        .contiguous();

    // Contract the shared rank dimension via Batched Matrix Multiplication
    // [Batch, 1, Rank_Left] x [Batch, Rank_Left, Rank_Right] -> [Batch, 1,
    // Rank_Right]
    tmp = torch::bmm(tmp, sliced_core);
  }

  // Collapse the trailing singleton dimensions to return a flat batch
  // of scalars
  return tmp.squeeze({1, 2}); // Shape: [Batch]
}

// =================================================================
// Public operators

TTEngine TTEngine::operator-() const
{
  auto result = *this;
  result.cores_[0] = result.cores_[0].neg();
  return result;
}

TTEngine& TTEngine::operator+=(const TTEngine& other)
{
  int64_t num_cores = cores_.size();
  TORCH_CHECK(num_cores == other.cores_.size(),
    "TTEngines must have the same number of cores for addition.");

  if (num_cores == 1) {
    cores_[0] += other.cores_[0];
  }

  for (size_t i = 0; i < num_cores; ++i) {
    auto& core_a = cores_[i];
    const auto& core_b = other.cores_[i];

    auto shape_a = core_a.sizes();
    auto shape_b = core_b.sizes();

    // Validate physical dimensions (n and m)
    TORCH_CHECK(shape_a[1] == shape_b[1] && shape_a[2] == shape_b[2],
      "Physical dimensions must match at core ", i);

    int64_t ra_in = shape_a[0], ra_out = shape_a[3];
    int64_t rb_in = shape_b[0], rb_out = shape_b[3];
    int64_t n = shape_a[1], m = shape_a[2];

    // Determine new bond ranks
    // Boundary ranks stay at 1, internal ranks are summed
    int64_t new_r_in = (i == 0) ? 1 : (ra_in + rb_in);
    int64_t new_r_out = (i == num_cores - 1) ? 1 : (ra_out + rb_out);

    torch::Tensor new_core =
      torch::zeros({new_r_in, n, m, new_r_out}, core_a.options());

    // Place Core A and Core B into the new core
    if (i == 0) {
      // First Core: Horizontal Concatenation [A, B]
      new_core.slice(0, 0, 1).slice(3, 0, ra_out) = core_a;
      new_core.slice(0, 0, 1).slice(3, ra_out, ra_out + rb_out) = core_b;
    } else if (i == num_cores - 1) {
      // Last Core: Vertical Concatenation [A; B]
      new_core.slice(0, 0, ra_in).slice(3, 0, 1) = core_a;
      new_core.slice(0, ra_in, ra_in + rb_in).slice(3, 0, 1) = core_b;
    } else {
      // Internal Cores: Block Diagonal [A 0; 0 B]
      new_core.slice(0, 0, ra_in).slice(3, 0, ra_out) = core_a;
      new_core.slice(0, ra_in, ra_in + rb_in)
        .slice(3, ra_out, ra_out + rb_out) = core_b;
    }

    core_a = new_core;
  }

  return *this;
}
TTEngine& TTEngine::operator-=(const TTEngine& other)
{
  return *this += (-other);
}
TTEngine& TTEngine::operator*=(const TTEngine& other)
{
  int64_t num_cores = cores_.size();
  TORCH_CHECK(num_cores == other.cores_.size(),
    "TTEngines must have the same number of cores for addition.");

  for (size_t i = 0; i < num_cores; ++i) {
    auto& core_a = cores_[i];
    const auto& core_b = other.cores_[i];

    // Verify physical dimensions match
    TORCH_CHECK(
      core_a.size(1) == core_b.size(1) && core_a.size(2) == core_b.size(2),
      "Physical dimensions must match for Hadamard product.");

    // Use einsum to perform the Kronecker-like product on ranks
    // a: [r1_in, n, m, r1_out]
    // b: [r2_in, n, m, r2_out]
    // We multiply matching n, m and combine the r_in and r_out dimensions
    // Result: [r1_in, r2_in, n, m, r1_out, r2_out]
    torch::Tensor combined =
      torch::einsum("unmv,wnmz-> uwnmvz", {core_a, core_b});

    // Reshape to combine the bond ranks back into 4D
    core_a = combined.reshape({core_a.size(0) * core_b.size(0), core_a.size(1),
      core_a.size(2), core_a.size(3) * core_b.size(3)});
  }

  return *this;
}

TTEngine& TTEngine::operator/=(const TTEngine& other)
{
  *this = elementwise_divide(*this, other);
  return *this;
}

template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value ||
                          std::is_same<T, torch::Tensor>::value,
  TTEngine>::type
operator/(const T& a, const TTEngine& b)
{
  // Create a numerator out of a
  auto numerator = TTEngine::ones(
    b.get_m_modes(), b.get_n_modes(), b.get_device(), b.get_dtype());
  numerator *= a;
  return elementwise_divide(numerator, b);
}

TTEngine operator/(const TTEngine& a, const TTEngine& b)
{
  return elementwise_divide(a, b);
}

template TTEngine operator/ <float>(const float& a, const TTEngine& b);
template TTEngine operator/ <double>(const double& a, const TTEngine& b);
template TTEngine operator/
  <torch::Tensor>(const torch::Tensor& a, const TTEngine& b);

// =================================================================
// Public getters / setters
int64_t TTEngine::get_numel() const
{
  int64_t total = 0;
  for (const auto& core : cores_) {
    total += core.numel();
  }
  return total;
}

c10::SmallVector<int64_t, 7> TTEngine::get_ranks() const
{
  c10::SmallVector<int64_t, 7> ranks;
  ranks.reserve(cores_.size() - 1);

  for (auto it = cores_.begin(); it != cores_.end(); it++) {
    ranks.push_back(it->size(0));
  }
  ranks.push_back(1);

  return ranks;
}

c10::SmallVector<int64_t, 12> TTEngine::get_free_indices() const
{
  c10::SmallVector<int64_t, 12> free_indices;
  free_indices.reserve(2 * cores_.size());

  for (const auto& core : cores_) {
    free_indices.push_back(core.size(1));
    free_indices.push_back(core.size(2));
  }

  return free_indices;
}

c10::SmallVector<int64_t, 6> TTEngine::get_m_modes() const
{
  c10::SmallVector<int64_t, 6> m_modes;
  m_modes.reserve(cores_.size());

  for (const auto& core : cores_) {
    m_modes.push_back(core.size(1));
  }

  return m_modes;
}

c10::SmallVector<int64_t, 6> TTEngine::get_n_modes() const
{
  c10::SmallVector<int64_t, 6> n_modes;
  n_modes.reserve(cores_.size());

  for (const auto& core : cores_) {
    n_modes.push_back(core.size(2));
  }

  return n_modes;
}

} // namespace ttnte::linalg
