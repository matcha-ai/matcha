#pragma once

#include "bits_of_matcha/engine/op/Op.h"


namespace matcha::engine::ops {

struct Require : Op {
  explicit Require(Tensor* a);
  explicit Require(const std::vector<Tensor*>& tensors);
  explicit Require(std::initializer_list<Tensor*> tensors);
  static OpMeta<Require> meta;
};

}