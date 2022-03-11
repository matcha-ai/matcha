#pragma once

#include "bits_of_matcha/engine/fn.h"


namespace matcha::fn {
Tensor dot(const Tensor& a, const Tensor& b);
}


namespace matcha::engine::fn {

class Dot: public Node {
  public:
    Dot(const matcha::Tensor* a, const matcha::Tensor* b);
    Dot(Tensor* a, Tensor* b);

    void run() override;
    void use(const Device& device) override;
    const Device::Concrete* device() const override;

  private:
    MatrixStackIteration a_, b_;
    Device::Concrete dev_;
};

}