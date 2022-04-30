#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinary.h"

#include <algorithm>
#include <numeric>
#include <execution>
#include <iostream>


namespace matcha::engine {


struct ElementwiseBinaryOp : Op {
  ElementwiseBinaryOp(Tensor* a, Tensor* b)
    : Op{a, b}
    , ctx_(a->shape(), b->shape())
  {
    if (a->dtype() != b->dtype()) throw std::invalid_argument("dtype mismatch");
    outputs.add(this, a->dtype(), ctx_.dimsC);
  }

protected:
  ElementwiseBinaryCtx ctx_;

  template <class Callable>
  void runCPU(Callable callable) {
    cpu::elementwiseBinary(
      callable,
      inputs[0]->buffer(),
      inputs[1]->buffer(),
      outputs[0]->malloc(),
      ctx_
    );
  };

};




}