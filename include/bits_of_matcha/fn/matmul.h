#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor matmul(const Tensor& a, const Tensor& b);
  UnaryFn matmulWith(const Tensor& b);
  UnaryFn matmulAgainst(const Tensor& a);
}
}


namespace matcha {
namespace engine {
namespace fn {


class Matmul : public Fn {
  public:
    Matmul(Tensor* a, Tensor* b);
    Matmul(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
