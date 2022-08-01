#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/op/OpBack.h"


namespace matcha::engine::ops {

struct Identity final : Op {
  explicit Identity(Tensor* a);
  explicit Identity(Tensor* a, Tensor* target);
  static Reflection<Identity> reflection;

  void run() override;
};

}
