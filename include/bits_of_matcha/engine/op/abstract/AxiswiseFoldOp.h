#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/AxiswiseFoldCtx.h"


namespace matcha::engine {

struct FoldOp : Op {
  FoldOp(Tensor* a)
    : Op{a}
    , ctx_(a->shape())
  {}

  FoldOp(Tensor* a, int axis)
    : Op{a}
    , ctx_(a->shape(), axis)
  {}

protected:
  AxiswiseFoldCtx ctx_;
};

}