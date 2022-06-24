#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Gt : ElementwiseBinaryOp {
  Gt(Tensor* a, Tensor* b);
  static OpMeta<Gt> meta;

  void run() override;
};

}
