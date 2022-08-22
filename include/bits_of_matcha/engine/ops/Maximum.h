#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOpBack.h"


namespace matcha::engine::ops {

struct Maximum : ElementwiseBinaryOp {
  Maximum(Tensor* a, Tensor* b);
  static Reflection<Maximum> reflection;

  void run() override;
};

struct MaximumBack : ElementwiseBinaryOpBack {
  MaximumBack(const BackCtx& ctx);
  static Reflection<MaximumBack> reflection;

  void run() override;
};


}
