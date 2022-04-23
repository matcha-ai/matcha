#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/autograd/OpBack.h"


namespace matcha::engine::ops {

struct Add : ElementwiseBinaryOp {
  Add(Tensor* a, Tensor* b);
  static OpMeta<Add> meta;

  void run() override;
};

struct AddBack : OpBack {
  AddBack(const BackCtx& ctx);
  static OpMeta<AddBack> meta;

};


}