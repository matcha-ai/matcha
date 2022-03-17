#pragma once

#include "bits_of_matcha/engine/fn.h"


namespace matcha::fn {

tensor transpose(const tensor& a);

}


namespace matcha::engine::fn {

class Transpose : public Node  {
public:
  Transpose(Tensor* a);

  void init() override;
  void run() override;
  void use(const Device& device) override;
  const Device::Concrete* device() const override;

private:
  MatrixStackIteration a_;
  Device::Concrete dev_;
  bool idle_;
};

}