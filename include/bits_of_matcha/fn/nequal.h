#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor nequal(const Tensor& a, const Tensor& b);
  UnaryFn nequalWith(const Tensor& b);
  UnaryFn nequalAgainst(const Tensor& a);
}
}


matcha::Tensor operator!=(const matcha::Tensor& a, const matcha::Tensor& b);


namespace matcha {
namespace engine {
namespace fn {


class Nequal : public Fn {
  public:
    Nequal(Tensor* a, Tensor* b);
    Nequal(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
