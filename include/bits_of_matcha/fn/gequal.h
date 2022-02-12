#pragma once

#include "bits_of_matcha/tensor.h"

namespace matcha {
namespace fn {
  Tensor gequal(const Tensor& a, const Tensor& b);
}
}

matcha::Tensor operator>=(const matcha::Tensor& a, const matcha::Tensor& b);
