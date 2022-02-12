#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor land(const Tensor& a, const Tensor& b);
}
}


matcha::Tensor operator&&(const matcha::Tensor& a, const matcha::Tensor& b);


namespace matcha {
namespace engine {
namespace fn {


class Land : public Fn {
  public:
    Land(Tensor* a, Tensor* b);
    Land(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
