#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"

namespace matcha::engine::ops {

struct Assign : Op {
  explicit Assign(Tensor* target, Tensor* source);
  static OpMeta<Assign> meta;

  void run() override;

private:
  ElementwiseBinaryCtx iter_;
};

}