#pragma once

#include "bits_of_matcha/Frame.h"

#include <vector>
#include <array>


namespace matcha::engine {

struct AxiswiseFoldCtx {
  AxiswiseFoldCtx(const Shape& a);
  AxiswiseFoldCtx(const Shape& a, int axis);
  AxiswiseFoldCtx() = default;

  size_t axisStride;
  size_t axisLength;

  std::array<size_t, 2> prefixStrides;
  std::array<size_t, 2> prefixLengths;
};

}