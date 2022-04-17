#pragma once

#include "bits_of_matcha/engine/op/Op.h"
#include "bits_of_matcha/engine/tensor/iterations.h"

#include <algorithm>
#include <numeric>
#include <execution>
#include <iostream>


namespace matcha::engine {


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
  ElementwiseBinaryIteration iter_;
};




}