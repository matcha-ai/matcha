#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryLogicalOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Gt : ElementwiseBinaryLogicalOp {
  Gt(Tensor* a, Tensor* b);
  static Reflection<Gt> reflection;

  void run() override;
};

}
