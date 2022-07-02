#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryLogicalOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Eq : ElementwiseBinaryLogicalOp {
  Eq(Tensor* a, Tensor* b);
  static OpMeta<Eq> meta;

  void run() override;
};

}
