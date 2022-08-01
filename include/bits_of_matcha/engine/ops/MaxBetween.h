#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOpBack.h"


namespace matcha::engine::ops {

struct MaxBetween : ElementwiseBinaryOp {
  MaxBetween(Tensor* a, Tensor* b);
  static Reflection<MaxBetween> reflection;

  void run() override;
};

struct MaxBetweenBack : ElementwiseBinaryOpBack {
  MaxBetweenBack(const BackCtx& ctx);
  static Reflection<MaxBetweenBack> reflection;

  void run() override;
};


}
