#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/nodeloader.h"


namespace matcha {
class Tensor;
class Stream;
class Tuple;

namespace fn {
  Tensor equal(const Tensor& a, const Tensor& b);
}
}

matcha::Tensor operator==(const matcha::Tensor& a, const matcha::Tensor& b);

namespace matcha {
namespace device {
  class Computation;
}

namespace engine {
class Tensor;

namespace fn {


class Equal : public Fn {
  public:
    Equal(Tensor* a, Tensor* b);
    Equal(const matcha::Tensor& a, const matcha::Tensor& b);
};


}
}
}
