#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor sqrt(const Tensor& a);
}
}


namespace matcha {
namespace engine {
namespace fn {


class Sqrt : public Fn {
  public:
    Sqrt(Tensor* a);
    Sqrt(const matcha::Tensor& a);
};


}
}
}
