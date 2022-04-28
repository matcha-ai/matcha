#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Divide : ElementwiseBinaryOp {
  Divide(Tensor* a, Tensor* b);
  static OpMeta<Divide> meta;

  void run() override;
};

struct DivideBack : OpBack {
  DivideBack(const BackCtx& ctx);
  static OpMeta<DivideBack> meta;

  void run() override;

protected:
  ElementwiseBinaryCtx iter_;
};


}
