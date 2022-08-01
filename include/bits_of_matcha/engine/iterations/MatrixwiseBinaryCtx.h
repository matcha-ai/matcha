#pragma once

#include "bits_of_matcha/Frame.h"

#include <vector>
#include <stdexcept>

namespace matcha::engine {

struct MatrixwiseBinaryCtx {
  MatrixwiseBinaryCtx(const Shape& a, const Shape& b);
  MatrixwiseBinaryCtx() = default;

  std::vector<unsigned> prefix_dims_c;
  std::vector<size_t> prefix_strides_a;
  std::vector<size_t> prefix_strides_b;
  std::vector<size_t> prefix_strides_c;
  unsigned rows_a, rows_b;
  unsigned cols_a, cols_b;
};

}