#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Eq : ElementwiseBinaryOp {
  Eq(Tensor* a, Tensor* b);
  static OpMeta<Eq> meta;

  void run() override;
};

}
