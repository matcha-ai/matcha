#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/nodeloader.h"


namespace matcha {
class Tensor;

namespace fn {
  Tensor square(const Tensor& a);
}
}

namespace matcha {
namespace device {
  class Computation;
}

namespace engine {
class Tensor;

namespace fn {


class Square : public Fn {
  public:
    Square(Tensor* a);
    Square(const matcha::Tensor& a);
};


}
}
}
