#pragma once

#include "bits_of_matcha/engine/memory/Buffer.h"

#include <algorithm>
#include <execution>

namespace matcha::engine::cpu {

template <class T>
void fill(engine::Buffer* buffer, size_t size, T value) {
  auto vals = buffer->as<T*>();
  std::fill(std::execution::par_unseq, vals, vals + size, value);
}

}