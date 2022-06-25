#pragma once

#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine::ops {

struct Cast : Op {
  Cast(Tensor* a, const Dtype& dtype);
  static OpMeta<Cast> meta;

  void run() override;
};

}
