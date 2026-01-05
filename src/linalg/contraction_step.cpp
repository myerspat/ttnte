#include "ttnte/linalg/contraction_step.hpp"
#include <algorithm>

namespace ttnte::linalg {
// =================================================
// Public constructors
// =================================================
ContractionStep::ContractionStep(const int64_t& lndim, const int64_t& rndim,
  const std::vector<int64_t>& ldims, const std::vector<int64_t>& rdims,
  const std::optional<std::vector<int64_t>>& permute)
  : permute_(permute)
{
  assert(ldims.size() == rdims.size());
  assert(*std::max_element(ldims.cbegin(), ldims.cend()) < lndim);
  assert(*std::max_element(rdims.cbegin(), rdims.cend()) < rndim);

  // Check if contraction indices are sorted
  use_tensordot_ = (std::is_sorted(ldims.cbegin(), ldims.cend()) &&
                    std::is_sorted(rdims.cbegin(), rdims.cend()));

  if (!use_tensordot_) {
    // Generate einsum expression
    einsum_expr_ = generate_expression(lndim, rndim, ldims, rdims);

  } else {
    ldims_ = ldims;
    rdims_ = rdims;
  }
}

// =================================================
// Private methods
// =================================================
std::string ContractionStep::generate_expression(const int64_t& lndim,
  const int64_t& rndim, const std::vector<int64_t>& ldims,
  const std::vector<int64_t>& rdims) const
{
  // String to iterate through
  std::string iter_str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::map<int, char> cdims;
  std::string in;
  std::string out;

  // Check length
  assert((lndim + rndim - ldims.size()) <= iter_str.size());

  // Handle first tensor
  int i = 0;
  for (int j = 0; j < lndim; j++) {
    in += iter_str[j];

    // Check if character is for a dimension that contracts
    if (ldims[i] == j) {
      cdims.insert({rdims[i], iter_str[j]});
      i++;
    } else {
      out += iter_str[j];
    }
  }
  in += ",";

  // Handle second tensor
  for (int j = 0; j < rndim; j++) {
    if (cdims.find(j) == cdims.end()) {
      in += iter_str[j + lndim];
      out += iter_str[j + lndim];
    } else {
      in += cdims.at(j);
    }
  }

  return in + "->" + out;
}

// =================================================
// Public methods
// =================================================
torch::Tensor ContractionStep::contract(
  const torch::Tensor& ltensor, const torch::Tensor& rtensor) const
{
  // Apply contraction
  torch::Tensor result = use_tensordot_
                           ? torch::tensordot(ltensor, rtensor, ldims_, rdims_)
                           : torch::einsum(einsum_expr_, {ltensor, rtensor});

  // Apply permutation
  return permute_.has_value()
           ? torch::permute(result, permute_.value()).contiguous()
           : result;
}

} // namespace ttnte::linalg
