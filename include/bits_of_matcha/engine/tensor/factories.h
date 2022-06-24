#pragma once

#include "bits_of_matcha/engine/tensor/Tensor.h"


namespace matcha::engine {

template <class Function>
Tensor* generate(const Function& function, const Shape& shape) {
  auto tensor = new Tensor(Float, shape);
  auto values = tensor->malloc()->as<float*>();
  for (size_t i = 0; i < tensor->size(); i++) {
    values[i] = function();
  }
  return tensor;
}

Tensor* full(float value, const Shape& shape);
Tensor* zeros(const Shape& shape);
Tensor* ones(const Shape& shape);
Tensor* eye(const Shape& shape);

Tensor* blob(const void* data, const Frame& frame);

}