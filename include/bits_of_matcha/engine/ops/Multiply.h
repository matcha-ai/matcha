#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOpBack.h"


namespace matcha::engine::ops {

struct Multiply : ElementwiseBinaryOp {
  Multiply(Tensor* a, Tensor* b);
  static Reflection<Multiply> reflection;

  void run() override;
};

struct MultiplyBack : ElementwiseBinaryOpBack {
  MultiplyBack(const BackCtx& ctx);
  static Reflection<MultiplyBack> reflection;

  void run() override;
};


}
