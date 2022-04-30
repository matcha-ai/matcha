#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct MinBetween : ElementwiseBinaryOp {
  MinBetween(Tensor* a, Tensor* b);
  static OpMeta<MinBetween> meta;

  void run() override;
};

struct MinBetweenBack : OpBack {
  MinBetweenBack(const BackCtx& ctx);
  static OpMeta<MinBetweenBack> meta;

  void run() override;

protected:
  ElementwiseBinaryCtx iter_;
};


}
