#pragma once

#include <memory>
#include <torch/extension.h>

namespace ttnte::linalg {

class Operator : public std::enable_shared_from_this<Operator> {
public:
  // =================================================
  // Public methods
  virtual ~Operator() = default;

  virtual torch::Tensor apply(const torch::Tensor& x) const = 0;
  virtual void cuda(const int64_t idx) = 0;
  virtual void cpu() = 0;
  virtual void multiply(const double& other);

  // =================================================
  // Overloads
  std::shared_ptr<Operator> operator+(const std::shared_ptr<Operator>& other);
  std::shared_ptr<Operator> operator*(const double& other);

  template<typename Op>
  std::shared_ptr<Operator> operator-(const std::shared_ptr<Op>& other)
  {
    return operator+(std::make_shared<Op>(*other) * -1.0);
  };

  // =================================================
  // Getters / Setters
  virtual std::vector<int64_t> input_shape() const noexcept = 0;
  virtual std::vector<int64_t> output_shape() const noexcept = 0;
  virtual int64_t nelements() const noexcept = 0;
  virtual double compression() const noexcept = 0;
  std::shared_ptr<Operator> ptr() { return shared_from_this(); }
};

} // namespace ttnte::linalg
