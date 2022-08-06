#pragma once

#include "bits_of_matcha/engine/op/Op.h"

namespace matcha {
class tensor;
}

namespace matcha::engine::ops {

struct SideOutput : Op {
  explicit SideOutput(Tensor* source, tensor* target);
  static Reflection<SideOutput> reflection;

  void run() override;

  tensor* target();

private:
  tensor* target_;
};

}