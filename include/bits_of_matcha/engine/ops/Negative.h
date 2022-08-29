#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseUnaryOp.h"


namespace matcha::engine::ops {

struct Negative : ElementwiseUnaryOp {
  explicit Negative(Tensor* a);
  static Reflection<Negative> reflection;

  void run() override;
};

}
