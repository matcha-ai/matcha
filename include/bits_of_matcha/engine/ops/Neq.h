#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryLogicalOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Neq : ElementwiseBinaryLogicalOp {
  Neq(Tensor* a, Tensor* b);
  static Reflection<Neq> reflection;

  void run() override;
};

}
