#pragma once

#include "bits_of_matcha/engine/fn.h"

namespace matcha::fn {
tensor identity(const tensor& a);
}


namespace matcha::engine::fn {


class Identity : public Node {
public:
  Identity(const matcha::tensor* a);
  Identity(Tensor* a);

  void init() override;
  void run() override;

};


}