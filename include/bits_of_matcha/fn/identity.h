#pragma once

#include "bits_of_matcha/engine/fn.h"

namespace matcha::fn {
Tensor identity(const Tensor& a);
}


namespace matcha::engine::fn {


class Identity : public Node {
  public:
    Identity(const matcha::Tensor* a);
    Identity(Tensor* a);

    void init() override;
    void run() override;

};


}