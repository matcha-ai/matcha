#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/iterations/ElementwiseBinaryCtx.h"
#include "bits_of_matcha/engine/cpu/kernels/elementwiseBinary.h"

#include <algorithm>
#include <numeric>
#include <execution>
#include <iostream>


namespace matcha::engine {

/*

struct ElementwiseBinaryOp : Op {
  ElementwiseBinaryOp(Tensor* a, Tensor* b);
  ~ElementwiseBinaryOp();

  template <class BinaryOp>
  inline void runCPU(const BinaryOp& op) {
    auto a = inputs[0]->buffer()->as<float*>();
    auto b = inputs[1]->buffer()->as<float*>();
    auto c = outputs[0]->buffer()->as<float*>();

    switch (iter_.scalar) {
      default:

        std::transform(
          std::execution::par_unseq,
          a, a + iter_.size,
          b,
          c,
          op
        );
        break;

      case 0:

        std::transform(
          std::execution::par_unseq,
          b, b + iter_.size,
          c,
          [=](auto& b) { return op(a[0], b); }
        );
        break;

      case 1:
        std::transform(
          std::execution::par_unseq,
          a, a + iter_.size,
          c,
          [=](auto& a) { return op(a, b[0]); }
        );
        break;

    }

  }

protected:
  ElementwiseBinaryCtx iter_;
};

 */

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