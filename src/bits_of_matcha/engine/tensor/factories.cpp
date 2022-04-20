#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/engine/tensor/iterations.h"


namespace matcha::engine {

Tensor* full(float value, const Shape& shape) {
  auto tensor = new Tensor(Float, shape);

  auto buffer = tensor->malloc();
  auto floats = buffer->as<float*>();
  std::fill(floats, floats + tensor->size(), value);

  return tensor;
}

Tensor* zeros(const Shape& shape) {
  return full(0, shape);
}

Tensor* ones(const Shape& shape) {
  return full(1, shape);
}

Tensor* eye(const Shape& shape) {
  auto tensor = zeros(shape);
  auto floats = tensor->buffer()->as<float*>();

  auto iter = MatrixStackIteration(shape);
  for (int matrix = 0; matrix < iter.amount; matrix++) {
    for (int row = 0; row < std::min(iter.rows, iter.cols); row++) {
      floats[matrix * iter.size + row * (iter.cols + 1)] = 1;
    }
  }

  return tensor;
}

}