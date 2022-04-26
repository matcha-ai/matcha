#pragma once

#include "bits_of_matcha/engine/cpu/Buffer.h"

#include <execution>
#include <algorithm>
#include <numeric>


namespace matcha::engine::cpu {

template <class Callable>
void elementwiseUnary(Callable callable, engine::Buffer* a, engine::Buffer* b, size_t size) {
  auto valsA = a->as<float*>();
  auto valsB = b->as<float*>();

  std::transform(
    std::execution::par_unseq,
    valsA, valsA + size,
    valsB,
    callable
  );
}

}