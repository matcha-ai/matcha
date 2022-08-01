#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOpBack.h"


namespace matcha::engine::ops {

struct Add : ElementwiseBinaryOp {
  Add(Tensor* a, Tensor* b);
  static Reflection<Add> reflection;

  void run() override;
};

struct AddBack : ElementwiseBinaryOpBack {
  explicit AddBack(const BackCtx& ctx);
  static Reflection<AddBack> reflection;

  void run() override;
};


}