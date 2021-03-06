#pragma once

#include "bits_of_matcha/engine/memory/Buffer.h"

#include <execution>
#include <numeric>
#include <algorithm>
#include <complex>


namespace matcha::engine::cpu {

template <class InType, class OutType>
inline void cast(Buffer& in, Buffer& out, size_t size) {
  auto beginA = in.as<InType*>();
  auto endA = beginA + size;
  auto beginB = out.as<OutType*>();
  std::transform(
    std::execution::par_unseq,
    beginA, endA, beginB,
    [](InType x) {return (OutType) x;}
  );
}

template <class InType = std::complex<int32_t>, class OutType>
inline void ccast(Buffer& in, Buffer& out, size_t size) {
  auto beginA = in.as<InType*>();
  auto endA = beginA + size;
  auto beginB = out.as<OutType*>();
  std::transform(
    std::execution::par_unseq,
    beginA, endA, beginB,
    [](InType x) {return (OutType) x.real();}
  );
}

}