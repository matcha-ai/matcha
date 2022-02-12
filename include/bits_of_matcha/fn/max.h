#pragma once

#include "bits_of_matcha/tensor.h"


namespace matcha {
namespace fn {
  Tensor max(const Tensor& a);
  Tensor max(const Tensor& a, const Tensor& b);
}
}
