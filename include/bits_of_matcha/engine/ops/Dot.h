#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/op/OpBack.h"
#include "bits_of_matcha/engine/tensor/iterations.h"
#include "bits_of_matcha/engine/iterations/MatrixwiseBinaryCtx.h"


namespace matcha::engine::ops {

struct Dot : Op {
  Dot(Tensor* a, Tensor* b);
  static OpMeta<Dot> meta;

  void run() override;

private:
  MatrixwiseBinaryCtx iter_;
};

}