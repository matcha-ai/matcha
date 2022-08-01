#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Reshape : Op {
  Reshape(Tensor* a, const Shape::Reshape& dims);
  static Reflection<Reshape> reflection;

  void run() override;
};

}
