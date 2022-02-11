#pragma once

#include "bits_of_matcha/engine/node.h"


namespace matcha {
namespace engine {


class Inlet : public Node {
  public:
    Inlet(Tensor* input);

    const Dtype& dtype() const;
    const Shape& shape() const;
};


}
}
