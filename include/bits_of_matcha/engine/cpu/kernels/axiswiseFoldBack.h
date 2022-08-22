#pragma once

#include "bits_of_matcha/engine/memory/Block.h"
#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"


namespace matcha::engine::cpu {


template <class T, class Callable>
inline void axiswiseFoldBack(Callable callable, Buffer& a, Buffer& b, const AxiswiseFoldCtx& ctx) {
  auto vals_a = a.as<T*>();
  auto vals_b = b.as<T*>();

  T* beginsA[2] = {vals_a, vals_a};
  T* iter_b = vals_b;

  for (size_t iter0 = 0; iter0 < ctx.prefix_lengths[0]; iter0++) {
    beginsA[1] = beginsA[0];
    for (size_t iter1 = 0; iter1 < ctx.prefix_lengths[1]; iter1++) {
      T* beg = beginsA[1];
      T* end = beg + ctx.axis_stride * ctx.axis_length;
      callable(beg, ctx.axis_stride, end, *vals_b++);

      beginsA[1] += ctx.prefix_strides[1];
    }
    beginsA[0] += ctx.prefix_strides[0];
  }
}

}
