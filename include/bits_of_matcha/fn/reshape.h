#pragma once

#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/node.h"

#include <functional>


namespace matcha {
namespace fn {
  Tensor reshape(const Tensor& a, const Shape& shape);
  std::function<Tensor (const Tensor&)> reshape(const Shape& shape);
}
}


namespace matcha {
namespace engine {
namespace fn {


class Reshape : public Node {
  public:
    Reshape(Tensor* a, const Shape& shape);
    Reshape(const matcha::Tensor& a, const Shape& shape);

    void dataStatusChanged(In* in) override;
    void updateStatusChanged(In* in) override;
    void bufferChanged(In* in) override;

    void eval(Out* out) override;
    void prune(Out* out) override;

};


}
}
}
