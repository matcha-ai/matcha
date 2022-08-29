#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/op/OpBack.h"
#include "bits_of_matcha/engine/tensor/iterations.h"
#include "bits_of_matcha/engine/iterations/MatrixwiseBinaryCtx.h"


namespace matcha::engine::ops {

struct Matmul : Op {
  explicit Matmul(Tensor* a, Tensor* b,
                  std::pair<char, char> transpose = {'N', 'N'});
  static Reflection<Matmul> reflection;

  void run() override;

  std::pair<char, char> getTranspose() const;

private:
  MatrixwiseBinaryCtx iter_;
};

}