#pragma once

#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Argmin : AxiswiseFoldOp {
  Argmin(Tensor* a);
  Argmin(Tensor* a, int axis);
  static Reflection<Argmin> reflection;

  void run() override;
};

}
