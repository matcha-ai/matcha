#include "bits_of_matcha/engine/tensor/factories.h"
#include "bits_of_matcha/engine/tensor/iterations.h"
#include "bits_of_matcha/engine/cpu/kernels/fill.h"
#include "bits_of_matcha/print.h"

#include <execution>
#include <numeric>
#include <algorithm>
#include <cstring>


namespace matcha::engine {


Tensor* full(double value, const Shape& shape) {
  auto tensor = new Tensor(Double, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(float value, const Shape& shape) {
  auto tensor = new Tensor(Float, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(int8_t value, const Shape& shape) {
  auto tensor = new Tensor(Sbyte, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(int16_t value, const Shape& shape) {
  auto tensor = new Tensor(Short, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(int32_t value, const Shape& shape) {
  auto tensor = new Tensor(Int, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(int64_t value, const Shape& shape) {
  auto tensor = new Tensor(Long, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(uint8_t value, const Shape& shape) {
  auto tensor = new Tensor(Byte, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(uint16_t value, const Shape& shape) {
  auto tensor = new Tensor(Ushort, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(uint32_t value, const Shape& shape) {
  auto tensor = new Tensor(Uint, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(uint64_t value, const Shape& shape) {
  auto tensor = new Tensor(Ulong, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* full(bool value, const Shape& shape) {
  auto tensor = new Tensor(Bool, shape);
  cpu::fill(tensor->malloc(), tensor->size(), value);

  return tensor;
}

Tensor* zeros(const Shape& shape) {
  return engine::full((float) 0, shape);
}

Tensor* ones(const Shape& shape) {
  return engine::full((float) 1, shape);
}

Tensor* eye(const Shape& shape) {
  auto tensor = engine::zeros(shape);
  auto floats = tensor->buffer().as<float*>();

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
  auto b = t->malloc().as<uint8_t*>();
  memcpy(b, data, frame.bytes());
  return t;
}

}