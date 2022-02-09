#pragma once

#include "bits_of_matcha/engine/fn.h"

namespace matcha {
class Tensor;

namespace fn {

Tensor matmul(const Tensor& a, const Tensor& b);

}

namespace device {
  class Computation;
}

namespace engine {
class Tensor;

namespace fn {

class Matmul : public Fn {
  public:
    Matmul(Tensor* a, Tensor* b);
    Matmul(const matcha::Tensor& a, const matcha::Tensor& b);

    void eval(Tensor* target) override;

    static const NodeLoader* loader();
    const NodeLoader* getLoader() const override;

  private:
    device::Computation* computation_;
};

}
}
}
