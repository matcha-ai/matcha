#pragma once

#include "bits_of_matcha/tensor.h"


namespace matcha {
namespace fn {
  Tensor min(const Tensor& a);
  Tensor min(const Tensor& a, const Tensor& b);
}
}
