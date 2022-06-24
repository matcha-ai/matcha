#pragma once

#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine::ops {

struct Stack : Op {
  Stack(const std::vector<Tensor*>& inputs);

  void run() override;
};

}