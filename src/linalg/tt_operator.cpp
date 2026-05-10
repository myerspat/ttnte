#include "ttnte/linalg/tt_operator.hpp"

namespace ttnte::linalg {

// =================================================================
// Protected methods
Operator::Ptr TTOperator::to_impl(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return TTOperator::create(
    tt_matrix_.to(device, dtype, non_blocking, copy, memory_format), label_);
}

// =================================================================
// Public methods
TTOperator::Ptr TTOperator::clone_from(
  const TTEngine::Tensors& cores, std::optional<std::string> label)
{
  return TTOperator::create(TTEngine::clone_from(cores, true),
    label.has_value() ? Label::from_string(label.value())
                      : Label::create_internal());
}

TTOperator::Ptr TTOperator::clone_from(
  const TTEngine::Tensors& cores, const Label& label)
{
  return TTOperator::create(TTEngine::clone_from(cores, true), label);
}

TTOperator::Ptr TTOperator::to(const torch::Device& device,
  const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format) const
{
  return TTOperator::create(
    tt_matrix_.to(device, dtype, non_blocking, copy, memory_format), label_);
}

TTOperator::Ptr TTOperator::to(const at::ScalarType& dtype, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return TTOperator::create(
    tt_matrix_.to(get_device(), dtype, non_blocking, copy, memory_format),
    label_);
}

TTOperator::Ptr TTOperator::to(const torch::Device& device, bool non_blocking,
  bool copy, std::optional<at::MemoryFormat> memory_format) const
{
  return TTOperator::create(
    tt_matrix_.to(device, get_dtype(), non_blocking, copy, memory_format),
    label_);
}

void TTOperator::to_(const torch::Device& device, const at::ScalarType& dtype,
  bool non_blocking, bool copy, std::optional<at::MemoryFormat> memory_format)
{
  tt_matrix_.to_(device, dtype, non_blocking, copy, memory_format);
}

void TTOperator::to_(const at::ScalarType& dtype, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  tt_matrix_.to_(dtype, non_blocking, copy, memory_format);
};

void TTOperator::to_(const torch::Device& device, bool non_blocking, bool copy,
  std::optional<at::MemoryFormat> memory_format)
{
  tt_matrix_.to_(device, non_blocking, copy, memory_format);
};

void TTOperator::lr_orthogonalize_()
{
  tt_matrix_.lr_orthogonalize_();
}

TTOperator::Ptr TTOperator::lr_orthogonalize()
{
  auto copy = *this;
  copy.tt_matrix_.lr_orthogonalize_();
  copy.label_ = Label::create_internal();
  return std::make_shared<TTOperator>(copy);
}

void TTOperator::round_(double eps, int64_t max_rank)
{
  tt_matrix_.round_(eps, max_rank);
}

TTOperator::Ptr TTOperator::round(double eps, int64_t max_rank)
{
  auto copy = *this;
  copy.tt_matrix_.round_(eps, max_rank);
  copy.label_ = Label::create_internal();
  return std::make_shared<TTOperator>(copy);
}

TTOperator::Ptr TTOperator::from_dense(torch::Tensor tensor,
  const c10::SmallVector<int64_t, 6>& m_modes,
  const c10::SmallVector<int64_t, 6>& n_modes, double eps, int64_t max_rank,
  bool is_interleaved, std::optional<std::string> label)
{
  return create(TTEngine::from_dense(
                  tensor, m_modes, n_modes, eps, max_rank, is_interleaved),
    label);
}

torch::Tensor TTOperator::to_dense(bool interleave) const
{
  auto tensor = tt_matrix_.to_dense();

  if (!interleave) {
    size_t num_cores = tensor.ndimension() / 2;
    c10::SmallVector<int64_t, 12> dims(tensor.ndimension(), 0);

    for (size_t i = 0; i < num_cores; i++) {
      dims[i] = 2 * i;
      dims[i + num_cores] = dims[i] + 1;
    }

    tensor = tensor.permute(dims);
  }

  return tensor;
}

void TTOperator::from_buffer(const torch::Tensor& buffer)
{
  tt_matrix_.from_buffer(buffer);
}

void TTOperator::neg_()
{
  tt_matrix_.neg_();
}

TTOperator::Ptr TTOperator::zeros(const c10::SmallVector<int64_t, 6>& m_modes,
  const c10::SmallVector<int64_t, 6>& n_modes,
  std::optional<torch::Device> device, std::optional<torch::ScalarType> dtype)
{
  return TTOperator::create(TTEngine::zeros(m_modes, n_modes, device, dtype));
}

TTOperator::Ptr TTOperator::ones(const c10::SmallVector<int64_t, 6>& m_modes,
  const c10::SmallVector<int64_t, 6>& n_modes,
  std::optional<torch::Device> device, std::optional<torch::ScalarType> dtype)
{
  return TTOperator::create(TTEngine::ones(m_modes, n_modes, device, dtype));
}

void TTOperator::transpose_(const c10::SmallVector<int64_t, 6>& core_idxs)
{
  tt_matrix_.transpose_(core_idxs);
}

TTOperator::Ptr TTOperator::transpose(
  const c10::SmallVector<int64_t, 6>& core_idxs) const
{
  return TTOperator::create(tt_matrix_.transpose(core_idxs));
}

} // namespace ttnte::linalg
