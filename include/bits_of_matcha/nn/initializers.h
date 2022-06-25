#pragma once

#include "matcha/tensor"

#include <cmath>


namespace matcha::nn {

struct {
  Generator init() {
    auto g = [&] (const Shape& shape) {
      float range = std::sqrt((float) 6 / (float) (shape[-1] + shape[-2]));
      random::Uniform uni {-range, +range};
      return uni(shape);
    };
    return g;
  }

  MATCHA_GENERATOR_TAIL()
} glorot;

}