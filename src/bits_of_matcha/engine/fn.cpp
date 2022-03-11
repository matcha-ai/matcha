#include "bits_of_matcha/engine/fn.h"


namespace matcha::engine::fn {

MatrixStackIteration::MatrixStackIteration(const Shape& a) {
  switch (a.rank()) {
    case 0:
      throw std::invalid_argument("not a matrix (or stack of matrices)");
    case 1:
      rows = a.size();
      cols = 1;
      break;
    default:
      rows = a[-2];
      cols = a[-1];
  }
  size = rows * cols;
  amount = a.size() / size;
}

ElementwiseBinaryIteration::ElementwiseBinaryIteration(const Shape& a, const Shape& b) {
  if (!a.rank()) {
    size = b.size();
    scalar = 0;
    return;
  }
  if (!b.rank()) {
    size = a.size();
    scalar = 1;
    return;
  }
  if (a == b) {
    size = a.size();
    scalar = -1;
    return;
  }
  throw std::invalid_argument("incompatible shapes for elementwise binary operation");
}

}