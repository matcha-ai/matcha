#pragma once

#include "bits_of_matcha/engine/op/Op.h"

namespace matcha::engine::autograd {

struct AccumulateGrads : Op {
  AccumulateGrads(const std::vector<Tensor*>& grads, Tensor* target);
  static OpMeta<AccumulateGrads> meta;

  void run() override;
};

}