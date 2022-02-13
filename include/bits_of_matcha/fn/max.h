#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor max(const Tensor& a);
  Tensor max(const Tensor& a, const Tensor& b);
  UnaryFn maxWith(const Tensor& b);
  UnaryFn maxAgainst(const Tensor& a);
}
}
