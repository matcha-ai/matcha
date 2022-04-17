#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"


namespace matcha::engine::ops {

struct Add : ElementwiseBinaryOp {
  Add(Tensor* a, Tensor* b);
  static OpMeta<Add> meta;

  void run();
};

}