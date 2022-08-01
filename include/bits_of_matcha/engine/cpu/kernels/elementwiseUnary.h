#pragma once

#include "bits_of_matcha/engine/cpu/Buffer.h"

#include <execution>
#include <algorithm>
#include <numeric>


namespace matcha::engine::cpu {

template <class T, class Callable>
inline void elementwiseUnary(Callable callable, Buffer& a, Buffer& b, size_t size) {
//  return;
  auto vals_a = a.as<T*>();
  auto vals_b = b.as<T*>();

  std::transform(
    std::execution::par_unseq,
    vals_a, vals_a + size,
    vals_b,
    callable
  );
}

}