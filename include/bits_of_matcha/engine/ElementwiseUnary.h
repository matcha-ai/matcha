#pragma once

#include "bits_of_matcha/engine/fn.h"

#include <algorithm>
#include <numeric>
#include <execution>


namespace matcha::engine::fn {


class ElementwiseUnary : public Node {
public:
  ElementwiseUnary(Tensor* a);
  ~ElementwiseUnary();

  void use(const Device& device) override;
  const Device::Concrete* device() const override;

  template <class UnaryOp>
  inline void runCPU(const UnaryOp& op) {
    auto a = (float*) x_[0]->payload();
    auto b = (float*) y_[0]->payload();

    std::transform(
      std::execution::par_unseq,
      a, a + size_,
      b,
      op
    );
  }

protected:
  Device::Concrete dev_;
  size_t size_;

};




}
