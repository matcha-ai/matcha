#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/nodeloader.h"


namespace matcha {
class Tensor;
class Stream;
class Tuple;

namespace fn {
  Tensor sum(const Tensor& a);
}
}

namespace matcha {
namespace device {
  class Computation;
}

namespace engine {
class Tensor;

namespace fn {


class Sum : public Fn {
  public:
    Sum(Tensor* a);
    Sum(const matcha::Tensor& a);

    void eval(Tensor* target) override;

    static const NodeLoader* loader();
    const NodeLoader* getLoader() const override;

  private:
    device::Computation* computation_;
};


}
}
}
