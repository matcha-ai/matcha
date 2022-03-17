#pragma once

#include <cstddef>
#include <any>


namespace matcha {

struct Computation {
  enum {
    ElementwiseUnary,
    ElementwiseBinary,
    Fold,
    Dot,
    Transpose,
  } type;

  size_t cost;
};

}