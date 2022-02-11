#pragma once

#include "bits_of_matcha/engine/fn.h"


namespace matcha {

class Tensor;

namespace fn {
  Tensor greater(const Tensor& a, const Tensor& b);
}
}

matcha::Tensor operator>(const matcha::Tensor& a, const matcha::Tensor& b);

namespace matcha {
namespace engine {

class Tensor;

namespace fn {


class Greater : public Fn {
  public:
    Greater(Tensor* a, Tensor* b);
    Greater(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
