#pragma once

#include "bits_of_matcha/Frame.h"

#include <vector>
#include <stdexcept>

namespace matcha::engine {

class Tensor;

struct ElementwiseBinaryCtx {
  ElementwiseBinaryCtx(const Shape& a, const Shape& b);
  ElementwiseBinaryCtx() = default;

  std::vector<unsigned> dims_c;
  std::vector<unsigned> strides_a;
  std::vector<unsigned> strides_b;
  std::vector<unsigned> strides_c;
};


}