#pragma once

#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/engine/tensor/Buffer.h"

#include <algorithm>
#include <functional>
#include <execution>
#include <numeric>


namespace matcha::engine {
class Block;
class ElementwiseBinaryCtx;
}

namespace matcha::engine::cpu {

template <class T = float, class Callable>
inline void elementwiseBinaryLogical(Callable callable, Buffer& a, Buffer& b, Buffer& c, const ElementwiseBinaryCtx& ctx) {
//  return;

  auto vals_a = a.as<T*>();
  auto vals_b = b.as<T*>();
  auto vals_c = c.as<bool*>();

  int strides = (int) ctx.strides_a.size();
  if (strides == 1) {
    // scalars
    *vals_c = (bool) callable(*vals_a, *vals_b);
    return;
  }

  std::vector<unsigned> begin_a(strides, 0);
  std::vector<unsigned> begin_b(strides, 0);
  std::vector<unsigned> begin_c(strides, 0);
  auto iter_c = vals_c;

  int counter = 0;
  int axis = 1;
  while (true) {
//    if (counter++ > 20) exit(69);
//    print("==========");
//    print("axis ", axis, " begin_c ", begin_c[axis], " strides_c ", ctx.strides_c[axis - 1]);
//    print("axis: ", axis, " (strides ", strides, ")");
//    print("strides: ", ctx.strides_a[axis], " ", ctx.strides_b[axis], " -> ", ctx.strides_c[axis]);
//    print("begins: ", begin_a[axis], " ", begin_b[axis], " -> ", begin_c[axis]);
    if (axis != strides - 1) {
//      print("begin_c ", begin_c[axis]);
      if (begin_c[axis] == begin_c[axis - 1] + ctx.strides_c[axis - 1]) {
//        print("-> dec axis");
        if (axis == 1) break;
        axis--;
      } else {
        begin_a[axis + 1] = begin_a[axis];
        begin_b[axis + 1] = begin_b[axis];
        begin_c[axis + 1] = begin_c[axis];
        begin_a[axis] += ctx.strides_a[axis];
        begin_b[axis] += ctx.strides_b[axis];
        begin_c[axis] += ctx.strides_c[axis];
//        print("-> inc axis");
        axis++;
      }
    } else {
//      print("-> else");
      auto iA = vals_a + begin_a[axis];
      auto iB = vals_b + begin_b[axis];
      auto bC = vals_c + begin_c[axis];
      auto chunkSize = ctx.strides_c[axis - 1];
      auto eC = bC + chunkSize;
      auto sA = ctx.strides_a[axis];
      auto sB = ctx.strides_b[axis];

      for (auto iC = bC; iC != eC; iC++) {
//        print("compute ", std::distance(vals_a , iA), " ", std::distance(vals_b , iB), " -> ", std::distance(vals_c, iC));
        *iC = (bool) callable(*iA, *iB);
        iA += sA;
        iB += sB;
      }

      if (axis == 1) break;
      axis--;
    }

  }
}

}