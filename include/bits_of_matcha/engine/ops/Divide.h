#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOp.h"
#include "bits_of_matcha/engine/op/abstract/ElementwiseBinaryOpBack.h"


namespace matcha::engine::ops {

struct Divide : ElementwiseBinaryOp {
  explicit Divide(Tensor* a, Tensor* b);
  static OpMeta<Divide> meta;

  void run() override;
};

struct DivideBack : ElementwiseBinaryOpBack {
  explicit DivideBack(const BackCtx& ctx);
  static OpMeta<DivideBack> meta;

  void run() override;

};


}
