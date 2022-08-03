#pragma once

#include "bits_of_matcha/engine/op/abstract/AxiswiseFoldOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Max : AxiswiseFoldOp {
  explicit Max(Tensor* a, bool keep_dims);
  explicit Max(Tensor* a, int axis, bool keep_dims);
  static Reflection<Max> reflection;

  void run() override;
};

}