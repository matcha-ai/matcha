#pragma once

#include "bits_of_matcha/engine/op/abstract/ElementwiseUnaryOp.h"


namespace matcha::engine::ops {

struct Log : ElementwiseUnaryOp {
  explicit Log(Tensor* a);
  static Reflection<Log> reflection;

  void run() override;
};

}