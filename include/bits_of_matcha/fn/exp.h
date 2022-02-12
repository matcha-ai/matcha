#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor exp(const Tensor& a);
}
}


namespace matcha {
namespace engine {
namespace fn {


class Exp : public Fn {
  public:
    Exp(Tensor* a);
    Exp(const matcha::Tensor& a);
};


}
}
}
