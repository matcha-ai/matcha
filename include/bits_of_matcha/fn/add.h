#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor add(const Tensor& a, const Tensor& b);
  UnaryFn addWith(const Tensor& b);
  UnaryFn addAgainst(const Tensor& a);
}
}

matcha::Tensor operator+(const matcha::Tensor& a, const matcha::Tensor& b);
const matcha::Tensor& operator+=(matcha::Tensor& a, const matcha::Tensor& b);

namespace matcha {
namespace engine {
namespace fn {


class Add : public Fn {
  public:
    Add(Tensor* a, Tensor* b);
    Add(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
