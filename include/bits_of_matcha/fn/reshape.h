#pragma once

#include "bits_of_matcha/engine/node.h"


namespace matcha {

class Shape;
class Tensor;

namespace fn {

Tensor reshape(const Tensor& a, const Shape& shape);

}


namespace engine {

class Tensor;

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
