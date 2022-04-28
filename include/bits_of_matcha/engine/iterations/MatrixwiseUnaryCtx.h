#pragma once

#include "bits_of_matcha/Frame.h"

#include <vector>
#include <stdexcept>

namespace matcha::engine {

struct MatrixwiseUnaryCtx {
  MatrixwiseUnaryCtx(const Shape& a);

  unsigned rows, cols;
  size_t size, mats;
};

}
