#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/engine/tensor/iterations.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"
#include "bits_of_matcha/print.h"

#include <execution>
#include <numeric>
#include <algorithm>
#include <cstring>


namespace matcha::engine {

Tensor* full(float value, const Shape& shape) {
  auto tensor = new Tensor(Float, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* zeros(const Shape& shape) {
  return engine::full(0, shape);
}

Tensor* ones(const Shape& shape) {
  return engine::full(1, shape);
}

Tensor* eye(const Shape& shape) {
  auto tensor = engine::zeros(shape);
  auto floats = tensor->buffer()->as<float*>();

  auto iter = MatrixStackIteration(shape);
  for (int matrix = 0; matrix < iter.amount; matrix++) {
    for (int row = 0; row < std::min(iter.rows, iter.cols); row++) {
      floats[matrix * iter.size + row * (iter.cols + 1)] = 1;
    }
  }

  return tensor;
}

Tensor* blob(const void* data, const Frame& frame) {
  auto t = new Tensor(frame);
  auto b = t->malloc()->as<uint8_t*>();
  memcpy(b, data, frame.bytes());
  return t;
}

}