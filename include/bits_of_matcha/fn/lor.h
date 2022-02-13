#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor lor(const Tensor& a, const Tensor& b);
  UnaryFn lorWith(const Tensor& b);
  UnaryFn lorAgainst(const Tensor& a);
}
}


matcha::Tensor operator||(const matcha::Tensor& a, const matcha::Tensor& b);


namespace matcha {
namespace engine {
namespace fn {


class Lor : public Fn {
  public:
    Lor(Tensor* a, Tensor* b);
    Lor(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
