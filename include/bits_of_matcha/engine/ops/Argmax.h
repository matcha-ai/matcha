#pragma once

#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Argmax : AxiswiseFoldOp {
  Argmax(Tensor* a);
  Argmax(Tensor* a, int axis);
  static Reflection<Argmax> reflection;

  void run() override;
};

}
