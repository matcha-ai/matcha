#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Le : ElementwiseBinaryOp {
  Le(Tensor* a, Tensor* b);
  static OpMeta<Le> meta;

  void run() override;
};

}
