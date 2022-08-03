#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseUnaryOp.h"
#include "bits_of_matcha/engine/op/OpBack.h"

namespace matcha::engine::ops {

struct Exp : ElementwiseUnaryOp {
  explicit Exp(Tensor* a);
  static Reflection<Exp> reflection;

  void run() override;
};

}