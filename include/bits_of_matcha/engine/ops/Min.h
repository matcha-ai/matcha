#pragma once

#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Min : AxiswiseFoldOp {
  explicit Min(Tensor* a, bool keep_dims);
  explicit Min(Tensor* a, int axis, bool keep_dims);
  static Reflection<Min> reflection;

  void run() override;
};

}
