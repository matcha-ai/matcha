#pragma once

#include "bits_of_matcha/engine/fn.h"
#include "bits_of_matcha/engine/nodeloader.h"


namespace matcha {
class Tensor;
class Stream;
class Tuple;

namespace fn {
  Tensor add(const Tensor& a, const Tensor& b);
}
}

matcha::Tensor operator+(const matcha::Tensor& a, const matcha::Tensor& b);
const matcha::Tensor& operator+=(matcha::Tensor& a, const matcha::Tensor& b);

namespace matcha {
namespace device {
  class Computation;
}

namespace engine {
class Tensor;

namespace fn {


class Add : public Fn {
  public:
    Add(Tensor* a, Tensor* b);
    Add(const matcha::Tensor& a, const matcha::Tensor& b);

    void eval(Tensor* target) override;

    static const NodeLoader* loader();
    const NodeLoader* getLoader() const override;

  private:
    device::Computation* computation_;
};


}
}
}
