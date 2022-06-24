#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Lt : ElementwiseBinaryOp {
  Lt(Tensor* a, Tensor* b);
  static OpMeta<Lt> meta;

  void run() override;
};

}
