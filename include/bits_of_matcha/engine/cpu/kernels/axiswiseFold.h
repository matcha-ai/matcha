#pragma once

#include "bits_of_matcha/engine/memory/Block.h"
#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"


namespace matcha::engine::cpu {


template <class T, class Callable>
inline void axiswiseFold(Callable callable, Buffer& a, Buffer& b, const AxiswiseFoldCtx& ctx) {
  auto valsA = a.as<T*>();
  auto valsB = b.as<T*>();

  T* beginsA[2] = {valsA, valsA};
  T* iterB = valsB;

  for (size_t iter0 = 0; iter0 < ctx.prefixLengths[0]; iter0++) {
    beginsA[1] = beginsA[0];
    for (size_t iter1 = 0; iter1 < ctx.prefixLengths[1]; iter1++) {
      T* beg = beginsA[1];
      T* end = beg + ctx.axisStride * ctx.axisLength;
      *valsB++ = callable(beg, ctx.axisStride, end);

      beginsA[1] += ctx.prefixStrides[1];
    }
    beginsA[0] += ctx.prefixStrides[0];
  }
}

}