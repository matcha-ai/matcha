#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct MaxBetween : ElementwiseBinaryOp {
  MaxBetween(Tensor* a, Tensor* b);
  static OpMeta<MaxBetween> meta;

  void run() override;
};

struct MaxBetweenBack : OpBack {
  MaxBetweenBack(const BackCtx& ctx);
  static OpMeta<MaxBetweenBack> meta;

  void run() override;

protected:
  ElementwiseBinaryCtx iter_;
};


}
