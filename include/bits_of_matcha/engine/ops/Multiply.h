#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Multiply : ElementwiseBinaryOp {
  Multiply(Tensor* a, Tensor* b);
  static OpMeta<Multiply> meta;

  void run() override;
};

struct MultiplyBack : OpBack {
  MultiplyBack(const BackCtx& ctx);
  static OpMeta<MultiplyBack> meta;

  void run() override;

protected:
  ElementwiseBinaryCtx iter_;
};


}
