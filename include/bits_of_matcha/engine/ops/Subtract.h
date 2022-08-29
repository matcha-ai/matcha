#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOpBack.h"


namespace matcha::engine::ops {

struct Subtract : ElementwiseBinaryOp {
  Subtract(Tensor* a, Tensor* b);
  static Reflection<Subtract> reflection;

  void run() override;
};

struct SubtractBack : ElementwiseBinaryOpBack {
  explicit SubtractBack(const BackCtx& ctx);
  static Reflection<SubtractBack> reflection;

  void run() override;
};


}
