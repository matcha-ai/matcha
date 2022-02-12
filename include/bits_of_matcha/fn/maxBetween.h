#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor maxBetween(const Tensor& a, const Tensor& b);
}
}


namespace matcha {
namespace engine {
namespace fn {


class MaxBetween : public Fn {
  public:
    MaxBetween(Tensor* a, Tensor* b);
    MaxBetween(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
