#pragma once

#include "bits_of_matcha/engine/memory/Buffer.h"
#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"


namespace matcha::engine::cpu {


template <class Callable>
void axiswiseFold(Callable callable, engine::Buffer* a, engine::Buffer* b, const AxiswiseFoldCtx& ctx) {
  auto valsA = a->as<float*>();
  auto valsB = b->as<float*>();

  float* beginsA[2] = {valsA, valsA};
  float* iterB = valsB;

  for (size_t iter0 = 0; iter0 < ctx.prefixLengths[0]; iter0++) {
    beginsA[1] = beginsA[0];
    for (size_t iter1 = 0; iter1 < ctx.prefixLengths[1]; iter1++) {
      float* beg = beginsA[1];
      float* end = beg + ctx.axisStride * ctx.axisLength;
      *valsB++ = callable(beg, ctx.axisStride, end);

      beginsA[1] += ctx.prefixStrides[1];
    }
    beginsA[0] += ctx.prefixStrides[0];
  }
}

}