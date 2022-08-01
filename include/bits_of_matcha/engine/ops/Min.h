#pragma once

#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Min : AxiswiseFoldOp {
  Min(Tensor* a);
  Min(Tensor* a, int axis);
  static Reflection<Min> reflection;

  void run() override;
};

}
