#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Pow : ElementwiseBinaryOp {
  Pow(Tensor* a, Tensor* b);
  static OpMeta<Pow> meta;

  void run() override;
};

struct PowBack : OpBack {
  PowBack(const BackCtx& ctx);
  static OpMeta<PowBack> meta;

};


}
