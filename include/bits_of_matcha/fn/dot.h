#pragma once

#include "bits_of_matcha/engine/fn.h"


namespace matcha::fn {
tensor dot(const tensor& a, const tensor& b);
}


namespace matcha::engine::fn {

class Dot: public Node {
public:
  Dot(const matcha::tensor* a, const matcha::tensor* b);
  Dot(Tensor* a, Tensor* b);

  void run() override;
  void use(const Device& device) override;
  const Device::Concrete* device() const override;

private:
  MatrixStackIteration a_, b_;
  Device::Concrete dev_;
};

}