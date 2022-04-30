#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Neq : ElementwiseBinaryOp {
  Neq(Tensor* a, Tensor* b);
  static OpMeta<Neq> meta;

  void run() override;
};

}
