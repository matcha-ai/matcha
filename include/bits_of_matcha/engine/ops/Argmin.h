#pragma once

#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Argmin : AxiswiseFoldOp {
  explicit Argmin(Tensor* a, bool keep_dims);
  explicit Argmin(Tensor* a, int axis, bool keep_dims);
  static Reflection<Argmin> reflection;

  void run() override;
};

}
