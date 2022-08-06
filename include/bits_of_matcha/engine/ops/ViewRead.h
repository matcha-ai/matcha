#pragma once

#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine::ops {

struct ViewRead : Op {
  explicit ViewRead(engine::Tensor* source, const std::vector<engine::Tensor*>& idxs);
  static Reflection<ViewRead> reflection;

  void run() override;
};

}