#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor equal(const Tensor& a, const Tensor& b);
}
}


matcha::Tensor operator==(const matcha::Tensor& a, const matcha::Tensor& b);


namespace matcha {
namespace engine {
namespace fn {


class Equal : public Fn {
  public:
    Equal(Tensor* a, Tensor* b);
    Equal(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
