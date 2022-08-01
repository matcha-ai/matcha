#pragma once

#include "bits_of_matcha/engine/op/Op.h"

namespace matcha::engine {

struct AccumulateGrads : Op {
  AccumulateGrads(const std::vector<Tensor*>& grads);
  AccumulateGrads(const std::vector<Tensor*>& grads, Tensor* target);
  static Reflection<AccumulateGrads> reflection;

  void run();
};

}