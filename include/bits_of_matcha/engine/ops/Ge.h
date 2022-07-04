#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryLogicalOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Ge : ElementwiseBinaryLogicalOp {
  Ge(Tensor* a, Tensor* b);
  static OpMeta<Ge> meta;

  void run() override;
};

}
