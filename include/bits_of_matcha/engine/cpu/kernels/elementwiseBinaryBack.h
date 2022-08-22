#pragma once

#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/engine/cpu/Block.h"

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
inline void elementwiseBinaryBack(Callable callable, Buffer& a, Buffer& b, Buffer& c, const ElementwiseBinaryCtx& ctx) {

  const auto vals_a = a.as<T*>();
  const auto vals_b = b.as<T*>();
  const auto vals_c = c.as<T*>();

  const int strides = (int) ctx.strides_a.size();
  if (ctx.dims_c.empty()) {
    callable(*vals_a, *vals_b, *vals_c);
    return;
  }

  std::vector<T*> begin_a(strides, vals_a);
  std::vector<T*> begin_b(strides, vals_b);
  std::vector<T*> begin_c(strides, vals_c);

  begin_c.front() += ctx.strides_c.front();

  int counter = 0;
  int axis = 1;
  while (axis != 0) {
//    print("axis: ", axis);
    if (axis != strides - 1) {
//      std::cout << "begin_a:";
//      for (auto&& i: begin_a) std::cout << " " << i - vals_a;
//      std::cout << std::endl;
//      std::cout << "begin_b:";
//      for (auto&& i: begin_b) std::cout << " " << i - vals_b;
//      std::cout << std::endl;
//      std::cout << "begin_c:";
//      for (auto&& i: begin_c) std::cout << " " << i - vals_c;
//      std::cout << std::endl;

//      print(" chunk: ", ctx.strides_c[axis - 1]);

      if (begin_c[axis] == begin_c[axis - 1]) {
//        print("decreasing axis");
        axis--;
      } else {
        begin_a[axis + 1] = begin_a[axis];
        begin_b[axis + 1] = begin_b[axis];
        begin_c[axis + 1] = begin_c[axis];
        begin_a[axis] += ctx.strides_a[axis];
        begin_b[axis] += ctx.strides_b[axis];
        begin_c[axis] += ctx.strides_c[axis];
        axis++;
//        print("increasing axis");
      }
    } else {
//      print(begin_a[axis] - vals_a, " ", begin_b[axis] - vals_b, " ", begin_c[axis] - vals_c);
      auto iter_a = begin_a[axis];
      auto iter_b = begin_b[axis];
      auto iter_c = begin_c[axis];
      const auto end_c = begin_c[axis] + ctx.strides_c[axis - 1];
      const auto stride_a = ctx.strides_a[axis];
      const auto stride_b = ctx.strides_b[axis];

      while (iter_c != end_c) {
        callable(*iter_a, *iter_b, *iter_c);
        iter_a += stride_a;
        iter_b += stride_b;
        iter_c += 1;
      }

      axis--;
    }

  }
}

}
