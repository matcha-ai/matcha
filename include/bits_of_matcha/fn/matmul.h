#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor matmul(const Tensor& a, const Tensor& b);
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
