#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Minimum : ElementwiseBinaryOp {
  Minimum(Tensor* a, Tensor* b);
  static Reflection<Minimum> reflection;

  void run() override;
};

struct MinimumBack : OpBack {
  MinimumBack(const BackCtx& ctx);
  static Reflection<MinimumBack> reflection;

  void run() override;

protected:
  ElementwiseBinaryCtx iter_;
};


}
