#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Reshape : Op {
  Reshape(Tensor* a, const Shape& shape);
  static OpMeta<Reshape> meta;

  void run() override;
};

struct ReshapeBack : OpBack {
  ReshapeBack(const BackCtx& ctx);
  static OpMeta<ReshapeBack> meta;

  void run() override;
};

}
