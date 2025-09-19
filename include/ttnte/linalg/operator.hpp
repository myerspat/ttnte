#pragma once

#include <memory>
#include <torch/extension.h>

namespace ttnte::linalg {

class Operator : public std::enable_shared_from_this<Operator> {
private:
  // =================================================
  // Private data
  double scale_ = 1.0;

  // =================================================
  // Private methods

public:
  // =================================================
  // Public methods
  virtual ~Operator();

  virtual torch::Tensor apply(const torch::Tensor& x) const = 0;
  virtual void cuda(const int64_t idx) = 0;
  virtual void cpu() = 0;
  virtual std::shared_ptr<Operator> clone() const = 0;
  virtual std::shared_ptr<Operator> add_(
    const std::shared_ptr<Operator>& other) = 0;
  virtual std::shared_ptr<Operator> type(
    const caffe2::TypeMeta& dtype) const = 0;
  std::shared_ptr<Operator> add(const std::shared_ptr<Operator>& other) const;

  // =================================================
  // Overloads
  std::shared_ptr<Operator> operator+(const std::shared_ptr<Operator>& other);
  std::shared_ptr<Operator> operator-(const std::shared_ptr<Operator>& other);
  std::shared_ptr<Operator> operator-();

  // =================================================
  // Getters / Setters
  std::shared_ptr<Operator> ptr() { return shared_from_this(); }
  double scale() const noexcept { return scale_; }
  virtual void set_scale(const double& scale) { scale_ = scale; }
  virtual std::vector<int64_t> input_shape() const noexcept = 0;
  virtual std::vector<int64_t> output_shape() const noexcept = 0;
  virtual int64_t nelements() const noexcept = 0;
  virtual double compression() const noexcept = 0;
  virtual torch::Device device() const noexcept = 0;
  virtual caffe2::TypeMeta dtype() const noexcept = 0;
};

} // namespace ttnte::linalg
