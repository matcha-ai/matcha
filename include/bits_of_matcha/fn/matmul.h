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
};

}
}
}
