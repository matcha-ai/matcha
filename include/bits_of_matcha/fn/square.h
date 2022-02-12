#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/fn.h"


namespace matcha {
namespace fn {
  Tensor square(const Tensor& a);
}
}


namespace matcha {
namespace engine {
namespace fn {


class Square : public Fn {
  public:
    Square(Tensor* a);
    Square(const matcha::Tensor& a);
};


}
}
}
