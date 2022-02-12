#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor subtract(const Tensor& a, const Tensor& b);
}
}


matcha::Tensor operator-(const matcha::Tensor& a);
matcha::Tensor operator-(const matcha::Tensor& a, const matcha::Tensor& b);
const matcha::Tensor& operator-=(matcha::Tensor& a, const matcha::Tensor& b);


namespace matcha {
namespace engine {
namespace fn {


class Subtract : public Fn {
  public:
    Subtract(Tensor* a, Tensor* b);
    Subtract(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
