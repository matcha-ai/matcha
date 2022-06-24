#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Ge : ElementwiseBinaryOp {
  Ge(Tensor* a, Tensor* b);
  static OpMeta<Ge> meta;

  void run() override;
};

}
