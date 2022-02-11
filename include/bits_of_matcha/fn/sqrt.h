#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/nodeloader.h"


namespace matcha {
class Tensor;

namespace fn {
  Tensor sqrt(const Tensor& a);
}
}

namespace matcha {
namespace device {
  class Computation;
}

namespace engine {
class Tensor;

namespace fn {


class Sqrt : public Fn {
  public:
    Sqrt(Tensor* a);
    Sqrt(const matcha::Tensor& a);
};


}
}
}
