#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor min(const Tensor& a);
  Tensor min(const Tensor& a, const Tensor& b);
  UnaryFn minWith(const Tensor& b);
  UnaryFn minAgainst(const Tensor& a);
}
}
