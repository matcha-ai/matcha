#pragma once

#include "bits_of_matcha/engine/fn.h"

#include <algorithm>
#include <numeric>
#include <execution>
#include <iostream>


namespace matcha::engine::fn {


class ElementwiseBinary : public Node {
public:
  ElementwiseBinary(Tensor* a, Tensor* b);
  ~ElementwiseBinary();

  void use(const Device& device) override;
  const Device::Concrete* device() const override;

  template <class BinaryOp>
  inline void runCPU(const BinaryOp& op) {
    auto a = (float*) x_[0]->payload();
    auto b = (float*) x_[1]->payload();
    auto c = (float*) y_[0]->payload();

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
  Device::Concrete dev_;
  ElementwiseBinaryIteration iter_;
};




}