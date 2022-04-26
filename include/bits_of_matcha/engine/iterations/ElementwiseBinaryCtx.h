#pragma once

#include "bits_of_matcha/Frame.h"

#include <vector>
#include <stdexcept>

namespace matcha::engine {

class Tensor;

struct ElementwiseBinaryCtx {
  ElementwiseBinaryCtx(const Shape& a, const Shape& b);
  ElementwiseBinaryCtx() = default;

  std::vector<unsigned> dimsC;
  std::vector<unsigned> stridesA;
  std::vector<unsigned> stridesB;
  std::vector<unsigned> stridesC;
};


}