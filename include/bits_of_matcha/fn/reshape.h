#pragma once

#include "bits_of_matcha/engine/fn.h"


namespace matcha::fn {

tensor reshape(const tensor& a, const Shape::Reshape& shape);

}


namespace matcha::engine::fn {

class Reshape : public Node {
public:
  Reshape(Tensor* a, const Shape::Reshape& target);

  void init() override;
  void run() override;
};

}