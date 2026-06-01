#include "ttnte/linalg/tt_state.hpp"
#include "ttnte/parallel/boundary_buffer.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::linalg {

// =================================================================
// Protected methods
State::Ptr TTState::to_impl(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return TTState::create(
    tt_vector_.to(device, dtype, non_blocking, copy, memory_format), label_);
}

void TTState::check_cores() const
{
  const auto& cores = get_cores();

  for (const auto& core : cores) {
    if (core.size(2) != 1) {
      throw utils::runtime_error(*this, "ttnte::linalg::TTState::check_cores",
        "The second dimension is not size 1 for all cores");
    }
  }
}

// =================================================================
// Public methods
TTState::Ptr TTState::clone_from(
  const TTEngine::Tensors& cores, std::optional<std::string> label)
{
  return TTState::create(TTEngine::clone_from(cores, true),
    label.has_value() ? Label::from_string(label.value())
                      : Label::create_internal());
}

TTState::Ptr TTState::clone_from(
  const TTEngine::Tensors& cores, const Label& label)
{
  return TTState::create(TTEngine::clone_from(cores, true), label);
}

TTState::Ptr TTState::to(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return TTState::create(
    tt_vector_.to(device, dtype, non_blocking, copy, memory_format), label_);
}

TTState::Ptr TTState::to(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return TTState::create(
    tt_vector_.to(get_device(), dtype, non_blocking, copy, memory_format),
    label_);
}

TTState::Ptr TTState::to(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return TTState::create(
    tt_vector_.to(device, get_dtype(), non_blocking, copy, memory_format),
    label_);
}

void TTState::to_(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  tt_vector_.to_(device, dtype, non_blocking, copy, memory_format);
}

void TTState::to_(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  tt_vector_.to_(dtype, non_blocking, copy, memory_format);
};

void TTState::to_(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  tt_vector_.to_(device, non_blocking, copy, memory_format);
};

void TTState::lr_orthogonalize_()
{
  tt_vector_.lr_orthogonalize_();
}

TTState::Ptr TTState::lr_orthogonalize() const
{
  auto copy = *this;
  copy.tt_vector_.lr_orthogonalize_();
  copy.label_ = Label::create_internal();
  return std::make_shared<TTState>(copy);
}

void TTState::round_(double eps, int64_t max_rank)
{
  tt_vector_.round_(eps, max_rank);
}

TTState::Ptr TTState::round(double eps, int64_t max_rank) const
{
  auto copy = *this;
  copy.tt_vector_.round_(eps, max_rank);
  copy.label_ = Label::create_internal();
  return std::make_shared<TTState>(copy);
}

TTState::Ptr TTState::from_dense(const torch::Tensor& tensor, double eps,
  int64_t max_rank, std::optional<std::string> label)
{
  return create(TTEngine::from_dense(tensor, eps, max_rank), label);
}

torch::Tensor TTState::to_dense() const
{
  auto tensor = tt_vector_.to_dense(true);

  // Create a vector of dimensions to squeeze
  c10::SmallVector<int64_t, 6> dims;
  dims.reserve(tensor.ndimension() / 2);

  for (size_t i = 0; i < tensor.ndimension() / 2; i++) {
    dims.push_back(2 * i + 1);
  }
  tensor.squeeze_(dims);

  return tensor;
}

torch::Tensor TTState::pack() const
{
  // Get structural information
  const auto& cores = get_cores();
  const auto ranks = get_ranks();
  const auto free_indices = get_free_indices();

  // Get the number of cores, ranks, and dims
  const int64_t num_cores = static_cast<int64_t>(cores.size());
  const int64_t num_ranks = static_cast<int64_t>(ranks.size());
  const int64_t num_dims = static_cast<int64_t>(free_indices.size());

  // Collect the metadata into a single vector
  std::vector<int64_t> metadata = {
    static_cast<int64_t>(parallel::BoundaryType::TENSOR_TRAIN), num_cores,
    num_ranks, num_dims};
  metadata.insert(metadata.end(), ranks.begin(), ranks.end());
  metadata.insert(metadata.end(), free_indices.begin(), free_indices.end());

  // Convert metadata to a tensor with the same device/dtype as the cores
  auto options = torch::TensorOptions().dtype(get_dtype()).device(get_device());
  torch::Tensor buffer = torch::empty(
    {static_cast<int64_t>(metadata.size()) + get_numel()}, options);
  buffer.narrow(0, 0, metadata.size()).copy_(torch::tensor(metadata, options));

  // Fill the core data segments
  int64_t offset = metadata.size();
  for (const auto& core : cores) {
    const int64_t core_numel = core.numel();
    buffer.narrow(0, offset, core_numel).copy_(core.reshape({-1}));
    offset += core_numel;
  }

  return buffer;
}

TTState::Ptr TTState::unpack(const torch::Tensor& buffer, bool clone)
{
  // Get the metadata header information
  auto metadata_header = buffer.narrow(0, 0, 4).to(torch::kInt64).cpu();
  const int64_t* header_ptr = metadata_header.data_ptr<int64_t>();

  // Get the header information
  assert(header_ptr[0] ==
         static_cast<int64_t>(parallel::BoundaryType::TENSOR_TRAIN));
  int64_t num_cores = header_ptr[1];
  int64_t num_ranks = header_ptr[2];
  int64_t num_dims = header_ptr[3];

  // Extract ranks and free indices
  auto ranks = buffer.narrow(0, 4, num_ranks).to(torch::kInt64).cpu();
  auto free_indices =
    buffer.narrow(0, 4 + num_ranks, num_dims).to(torch::kInt64).cpu();

  // Slice out cores
  TTEngine::Tensors cores;
  cores.reserve(num_cores);
  int64_t offset = 4 + num_ranks + num_dims;

  auto ranks_ptr = ranks.data_ptr<int64_t>();
  auto free_indices_ptr = free_indices.data_ptr<int64_t>();

  for (int i = 0; i < num_cores; i++) {
    std::vector<int64_t> core_shape = {ranks_ptr[i], free_indices_ptr[i * 2],
      free_indices_ptr[i * 2 + 1], ranks_ptr[i + 1]};
    int64_t core_size =
      core_shape[0] * core_shape[1] * core_shape[2] * core_shape[3];

    cores.push_back(buffer.narrow(0, offset, core_size).reshape(core_shape));
    if (clone) {
      cores.back() = cores.back().clone();
    }
    offset += core_size;
  }

  return TTState::create(cores);
}

void TTState::neg_()
{
  tt_vector_.neg_();
}

TTState::Ptr TTState::zeros(const c10::SmallVector<int64_t, 6>& m_modes,
  std::optional<torch::Device> device, std::optional<torch::ScalarType> dtype)
{
  return TTState::create(TTEngine::zeros(m_modes, device, dtype));
}

TTState::Ptr TTState::ones(const c10::SmallVector<int64_t, 6>& m_modes,
  std::optional<torch::Device> device, std::optional<torch::ScalarType> dtype)
{
  return TTState::create(TTEngine::ones(m_modes, device, dtype));
}

TTOperator::Ptr TTState::transpose(
  const c10::SmallVector<int64_t, 6>& core_idxs) const
{
  return TTOperator::create(tt_vector_.transpose(core_idxs));
}

} // namespace ttnte::linalg
