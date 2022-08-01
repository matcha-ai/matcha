#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"


namespace matcha::engine::ops {

struct Sum : AxiswiseFoldOp {
  explicit Sum(Tensor* a);
  explicit Sum(Tensor* a, int axis);
  static Reflection<Sum> reflection;

  void run() override;
};

}
