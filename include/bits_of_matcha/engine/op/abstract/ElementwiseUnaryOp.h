#pragma once

#include "bits_of_matcha/engine/op/Op.h"

#include <algorithm>
#include <numeric>
#include <execution>


namespace matcha::engine {


struct ElementwiseUnaryOp : public Op {
  ElementwiseUnaryOp(Tensor* a);
  ~ElementwiseUnaryOp();

  template <class UnaryOp>
  inline void runCPU(const UnaryOp& op) {
    auto a = inputs[0]->buffer()->as<float*>();
    auto b = outputs[0]->buffer()->as<float*>();

    std::transform(
      std::execution::par_unseq,
      a, a + size_,
      b,
      op
    );
  }

protected:
  size_t size_;

};




}
