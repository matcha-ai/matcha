#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct MinBetween : ElementwiseBinaryOp {
  MinBetween(Tensor* a, Tensor* b);
  static Reflection<MinBetween> reflection;

  void run() override;
};

struct MinBetweenBack : OpBack {
  MinBetweenBack(const BackCtx& ctx);
  static Reflection<MinBetweenBack> reflection;

  void run() override;

protected:
  ElementwiseBinaryCtx iter_;
};


}
