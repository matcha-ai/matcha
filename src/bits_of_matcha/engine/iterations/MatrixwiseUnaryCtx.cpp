#include "bits_of_matcha/engine/iterations/MatrixwiseUnaryCtx.h"
#include "bits_of_matcha/error/IncompatibleShapesError.h"


namespace matcha::engine {

MatrixwiseUnaryCtx::MatrixwiseUnaryCtx(const Shape& a) {
  if (a.rank() < 2) {
    throw std::invalid_argument("expected at least a matrix (tensor with rank at least 2)");
  }

  cols = a[-1];
  rows = a[-2];
  size = rows * cols;
  mats = a.size() / size;
}

}