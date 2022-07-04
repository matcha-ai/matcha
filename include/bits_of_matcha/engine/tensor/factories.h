#pragma once

#include "bits_of_matcha/engine/tensor/Tensor.h"


namespace matcha::engine {

template <class Function>
Tensor* generate(const Function& function, const Shape& shape) {
  auto tensor = new Tensor(Float, shape);
  auto values = tensor->malloc().as<float*>();
  for (size_t i = 0; i < tensor->size(); i++) {
    values[i] = function();
  }
  return tensor;
}

Tensor* full(double value, const Shape& shape);
Tensor* full(float value, const Shape& shape);

Tensor* full(int8_t value, const Shape& shape);
Tensor* full(int16_t value, const Shape& shape);
Tensor* full(int32_t value, const Shape& shape);
Tensor* full(int64_t value, const Shape& shape);

Tensor* full(uint8_t value, const Shape& shape);
Tensor* full(uint16_t value, const Shape& shape);
Tensor* full(uint32_t value, const Shape& shape);
Tensor* full(uint64_t value, const Shape& shape);

Tensor* full(std::complex<int32_t> value, const Shape& shape);
Tensor* full(std::complex<uint32_t> value, const Shape& shape);
Tensor* full(std::complex<float> value, const Shape& shape);
Tensor* full(std::complex<double> value, const Shape& shape);

Tensor* full(bool value, const Shape& shape);

Tensor* zeros(const Shape& shape);
Tensor* ones(const Shape& shape);
Tensor* eye(const Shape& shape);

Tensor* blob(const void* data, const Frame& frame);

}