#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseUnaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"

namespace matcha::engine::ops {

struct Exp : ElementwiseUnaryOp {
  Exp(Tensor* a);
  static OpMeta<Exp> meta;

  void run() override;
};

struct ExpBack : OpBack {
  ExpBack(const BackCtx& ctx);
  static OpMeta<ExpBack> meta;

  void run() override;
};

}