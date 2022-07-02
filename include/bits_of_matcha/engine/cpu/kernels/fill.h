#pragma once

#include "bits_of_matcha/engine/memory/Block.h"

#include <algorithm>
#include <execution>

namespace matcha::engine::cpu {

template <class T>
void fill(Buffer& buffer, size_t size, T value) {
  auto vals = buffer.as<T*>();
  std::fill(std::execution::par_unseq, vals, vals + size, value);
}

}