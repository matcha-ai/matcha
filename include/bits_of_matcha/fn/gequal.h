#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/tensor.h"

namespace matcha {
namespace fn {
  Tensor gequal(const Tensor& a, const Tensor& b);
  UnaryFn gequalWith(const Tensor& b);
  UnaryFn gequalAgainst(const Tensor& a);
}
}

matcha::Tensor operator>=(const matcha::Tensor& a, const matcha::Tensor& b);
