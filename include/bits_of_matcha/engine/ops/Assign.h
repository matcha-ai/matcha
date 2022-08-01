#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"

namespace matcha::engine::ops {

struct Assign : Op {
  explicit Assign(Tensor* source, Tensor* target);
  Assign(const Assign& other);
  ~Assign();

  static Reflection<Assign> reflection;

  void run() override;

private:
  ElementwiseBinaryCtx iter_;
  Tensor* target_;
};

}