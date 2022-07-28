#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOpBack.h"


namespace matcha::engine::ops {

struct Add : ElementwiseBinaryOp {
  explicit Add(Tensor* a, Tensor* b);
  static OpMeta<Add> meta;

  void run() override;
};

struct AddBack : ElementwiseBinaryOpBack {
  explicit AddBack(const BackCtx& ctx);
  static OpMeta<AddBack> meta;

  void run() override;
};


}