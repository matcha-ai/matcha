#pragma once

#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/random.h"

#include <cmath>


namespace matcha::nn {

struct {
  Generator init() {
    auto g = [&] (const Shape& shape) {
      tensor range = std::sqrt(6. / (shape[-1] + shape[-2]));
      random::Uniform uni {-range, +range};
      return uni(shape);
    };
    return (Generator) g;
  }

  MATCHA_GENERATOR_TAIL()
} glorot;

}