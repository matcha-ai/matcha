#pragma once

#include "bits_of_matcha/Frame.h"

#include <vector>
#include <stdexcept>

namespace matcha::engine {

struct MatrixwiseBinaryCtx {
  MatrixwiseBinaryCtx(const Shape& a, const Shape& b);
  MatrixwiseBinaryCtx() = default;

  std::vector<unsigned> prefixDimsC;
  std::vector<size_t> prefixStridesA;
  std::vector<size_t> prefixStridesB;
  std::vector<size_t> prefixStridesC;
  unsigned rowsA, rowsB;
  unsigned colsA, colsB;
};

}