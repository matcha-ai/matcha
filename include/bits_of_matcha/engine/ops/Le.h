#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryLogicalOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Le : ElementwiseBinaryLogicalOp {
  Le(Tensor* a, Tensor* b);
  static Reflection<Le> reflection;

  void run() override;
};

}
