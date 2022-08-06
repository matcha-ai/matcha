#pragma once

#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine::ops {

struct ViewWrite : Op {
  explicit ViewWrite(engine::Tensor* source, engine::Tensor* rhs, const std::vector<engine::Tensor*>& idxs);
  static Reflection<ViewWrite> reflection;

  void run() override;
};

}
