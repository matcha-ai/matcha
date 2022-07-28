#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/op/OpBack.h"
#include "bits_of_matcha/engine/iterations/MatrixwiseUnaryCtx.h"


namespace matcha::engine::ops {

struct Transpose : Op {
  Transpose(Tensor* a);
  static OpMeta<Transpose> meta;

  void run() override;

private:
  MatrixwiseUnaryCtx iter_;
};

}
