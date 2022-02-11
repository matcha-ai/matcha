#pragma once

#include "bits_of_matcha/engine/fn.h"


namespace matcha {

class Tensor;

namespace fn {
  Tensor less(const Tensor& a, const Tensor& b);
}
}

matcha::Tensor operator<(const matcha::Tensor& a, const matcha::Tensor& b);

namespace matcha {
namespace engine {

class Tensor;

namespace fn {


class Less : public Fn {
  public:
    Less(Tensor* a, Tensor* b);
    Less(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
