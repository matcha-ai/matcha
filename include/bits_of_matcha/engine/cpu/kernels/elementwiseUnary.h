#pragma once

#include "bits_of_matcha/engine/cpu/Buffer.h"

#include <execution>
#include <algorithm>
#include <numeric>


namespace matcha::engine::cpu {

template <class T, class Callable>
inline void elementwiseUnary(Callable callable, Buffer& a, Buffer& b, size_t size) {
//  return;
  auto valsA = a.as<T*>();
  auto valsB = b.as<T*>();

  std::transform(
    std::execution::par_unseq,
    valsA, valsA + size,
    valsB,
    callable
  );
}

}