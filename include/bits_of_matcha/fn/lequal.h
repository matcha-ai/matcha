#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {

class Tensor;

namespace fn {
  Tensor lequal(const Tensor& a, const Tensor& b);
  UnaryFn lequalWith(const Tensor& b);
  UnaryFn lequalAgainst(const Tensor& a);
}
}

matcha::Tensor operator<=(const matcha::Tensor& a, const matcha::Tensor& b);
