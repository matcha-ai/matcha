#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseUnary.h"

#include <algorithm>
#include <numeric>
#include <execution>


namespace matcha::engine {


struct ElementwiseUnaryOp : public Op {
  ElementwiseUnaryOp(Tensor* a)
    : Op{a}
  {
    size_ = a->size();
    outputs.add(this, a->frame());
  }

  template <class Callable>
  inline void runCPU(const Callable& callable) {
    cpu::elementwiseUnary(
      callable,
      inputs[0]->buffer(),
      outputs[0]->malloc(),
      size_
    );
  }

protected:
  size_t size_;

};




}
