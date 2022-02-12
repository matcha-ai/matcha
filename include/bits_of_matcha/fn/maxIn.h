#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor maxIn(const Tensor& a);
}
}


namespace matcha {
namespace engine {
namespace fn {


class MaxIn : public Fn {
  public:
    MaxIn(Tensor* a);
    MaxIn(const matcha::Tensor& a);
};


}
}
}
