#pragma once

#include "bits_of_matcha/Frame.h"

#include <vector>
#include <array>


namespace matcha::engine {

struct AxiswiseFoldCtx {
  AxiswiseFoldCtx(const Shape& a, bool keep_dims);
  AxiswiseFoldCtx(const Shape& a, int axis, bool keep_dims);
  AxiswiseFoldCtx() = default;

  size_t axis_stride;
  size_t axis_length;

  std::array<size_t, 2> prefix_strides;
  std::array<size_t, 2> prefix_lengths;
};

}