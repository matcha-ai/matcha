#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryLogicalOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Lt : ElementwiseBinaryLogicalOp {
  Lt(Tensor* a, Tensor* b);
  static OpMeta<Lt> meta;

  void run() override;
};

}
