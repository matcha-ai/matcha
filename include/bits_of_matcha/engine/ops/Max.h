#pragma once

#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Max : AxiswiseFoldOp {
  Max(Tensor* a);
  Max(Tensor* a, int axis);
  static Reflection<Max> reflection;

  void run() override;
};

}