#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor greater(const Tensor& a, const Tensor& b);
  UnaryFn greaterWith(const Tensor& b);
  UnaryFn greaterAgainst(const Tensor& a);
}
}


matcha::Tensor operator>(const matcha::Tensor& a, const matcha::Tensor& b);


namespace matcha {
namespace engine {
namespace fn {


class Greater : public Fn {
  public:
    Greater(Tensor* a, Tensor* b);
    Greater(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
