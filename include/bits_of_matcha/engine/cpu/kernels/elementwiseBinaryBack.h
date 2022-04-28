#pragma once

#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/engine/cpu/Buffer.h"

#include <algorithm>
#include <functional>
#include <execution>
#include <numeric>


namespace matcha::engine {
class Buffer;
class ElementwiseBinaryCtx;
}

namespace matcha::engine::cpu {

template <class Callable>
inline void elementwiseBinaryBack(Callable callable, engine::Buffer* a, engine::Buffer* b, engine::Buffer* c, const ElementwiseBinaryCtx& ctx) {
//  return;
  dynamic_cast<Buffer*>(a);
  dynamic_cast<Buffer*>(b);
  dynamic_cast<Buffer*>(c);

  auto valsA = a->as<float*>();
  auto valsB = b->as<float*>();
  auto valsC = c->as<float*>();

  int strides = (int) ctx.stridesA.size();
  if (strides == 1) {
    // scalars
    callable(*valsA, *valsB, *valsC);
    return;
  }

  std::vector<unsigned> beginA(strides, 0);
  std::vector<unsigned> beginB(strides, 0);
  std::vector<unsigned> beginC(strides, 0);
  auto iterC = valsC;

  int counter = 0;
  int axis = 1;
  while (true) {
//    if (counter++ > 20) exit(69);
//    print("==========");
//    print("axis ", axis, " beginC ", beginC[axis], " stridesC ", ctx.stridesC[axis - 1]);
//    print("axis: ", axis, " (strides ", strides, ")");
//    print("strides: ", ctx.stridesA[axis], " ", ctx.stridesB[axis], " -> ", ctx.stridesC[axis]);
//    print("begins: ", beginA[axis], " ", beginB[axis], " -> ", beginC[axis]);
    if (axis != strides - 1) {
//      print("beginC ", beginC[axis]);
      if (beginC[axis] == beginC[axis - 1] + ctx.stridesC[axis - 1]) {
//        print("-> dec axis");
        if (axis == 1) break;
        axis--;
      } else {
        beginA[axis + 1] = beginA[axis];
        beginB[axis + 1] = beginB[axis];
        beginC[axis + 1] = beginC[axis];
        beginA[axis] += ctx.stridesA[axis];
        beginB[axis] += ctx.stridesB[axis];
        beginC[axis] += ctx.stridesC[axis];
//        print("-> inc axis");
        axis++;
      }
    } else {
//      print("-> else");
      auto iA = valsA + beginA[axis];
      auto iB = valsB + beginB[axis];
      auto bC = valsC + beginC[axis];
      auto chunkSize = ctx.stridesC[axis - 1];
      auto eC = bC + chunkSize;
      auto sA = ctx.stridesA[axis];
      auto sB = ctx.stridesB[axis];

      for (auto iC = bC; iC != eC; iC++) {
//        print("compute ", std::distance(valsA , iA), " ", std::distance(valsB , iB), " ", std::distance(valsC, iC));
        callable(*iA, *iB, *iC);
        iA += sA;
        iB += sB;
      }

      if (axis == 1) break;
      axis--;
    }

  }
}

}
