#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"


namespace matcha::engine::ops {

struct Broadcast : Op {
  explicit Broadcast(Tensor* a, const Shape& shape);
  static Reflection<Broadcast> reflection;

  void run() override;

private:
  ElementwiseBinaryCtx iter_;

};

}