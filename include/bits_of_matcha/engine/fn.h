#pragma once

#include "bits_of_matcha/engine/Node.h"


namespace matcha::engine::fn {

struct MatrixStackIteration {
  MatrixStackIteration(const Shape& a);

  unsigned rows, cols;
  size_t size;
  size_t amount;
};

struct ElementwiseBinaryIteration {
  ElementwiseBinaryIteration() = default;
  ElementwiseBinaryIteration(const Shape& a, const Shape& b);

  size_t size;
  int scalar;
};

}