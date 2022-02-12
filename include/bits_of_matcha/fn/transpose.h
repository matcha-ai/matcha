#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor transpose(const Tensor& a);
}
}


namespace matcha {
namespace engine {
namespace fn {


class Transpose : public Fn {
  public:
    Transpose(Tensor* a);
    Transpose(const matcha::Tensor& a);
};


}
}
}
