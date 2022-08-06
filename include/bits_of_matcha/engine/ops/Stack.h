#pragma once

#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine::ops {

struct Stack : Op {
  explicit Stack(const std::vector<Tensor*>& inputs);
  static Reflection<Stack> reflection;

  void run() override;
};

}