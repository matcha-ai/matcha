#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOpBack.h"


namespace matcha::engine::ops {

struct Power : ElementwiseBinaryOp {
  explicit Power(Tensor* a, Tensor* b);
  static Reflection<Power> reflection;

  void run() override;
};

struct PowerBack : ElementwiseBinaryOpBack {
  explicit PowerBack(const BackCtx& ctx);
  static Reflection<PowerBack> reflection;

  void run() override;
};


}
