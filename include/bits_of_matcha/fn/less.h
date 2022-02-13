#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor less(const Tensor& a, const Tensor& b);
  UnaryFn lessWith(const Tensor& b);
  UnaryFn lessAgainst(const Tensor& a);
}
}


matcha::Tensor operator<(const matcha::Tensor& a, const matcha::Tensor& b);


namespace matcha {
namespace engine {
namespace fn {


class Less : public Fn {
  public:
    Less(Tensor* a, Tensor* b);
    Less(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
