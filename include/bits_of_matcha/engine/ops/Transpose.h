#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/autograd/OpBack.h"
#include "bits_of_matcha/engine/tensor/iterations.h"


namespace matcha::engine::ops {

struct Transpose : Op {
  Transpose(Tensor* a);
  static OpMeta<Transpose> meta;

  void run() override;

private:
  MatrixStackIteration iter_;
};

struct TransposeBack : OpBack {
  TransposeBack(const BackCtx& ctx);
  static OpMeta<TransposeBack> meta;

  void run() override;
};


}
