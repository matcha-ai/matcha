#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/nodeloader.h"


namespace matcha {
class Tensor;

namespace fn {
  Tensor transpose(const Tensor& a);
}
}

namespace matcha {
namespace device {
  class Computation;
}

namespace engine {
class Tensor;

namespace fn {


class Transpose : public Fn {
  public:
    Transpose(Tensor* a);
    Transpose(const matcha::Tensor& a);
};


}
}
}
