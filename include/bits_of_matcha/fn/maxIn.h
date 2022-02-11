#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/nodeloader.h"


namespace matcha {
class Tensor;
class Stream;
class Tuple;

namespace fn {
  Tensor maxIn(const Tensor& a);
}
}

namespace matcha {
namespace device {
  class Computation;
}

namespace engine {
class Tensor;

namespace fn {


class MaxIn : public Fn {
  public:
    MaxIn(Tensor* a);
    MaxIn(const matcha::Tensor& a);
};


}
}
}
