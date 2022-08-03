#pragma once

#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Argmax : AxiswiseFoldOp {
  explicit Argmax(Tensor* a, bool keep_dims);
  explicit Argmax(Tensor* a, int axis, bool keep_dims);
  static Reflection<Argmax> reflection;

  void run() override;
};

}
