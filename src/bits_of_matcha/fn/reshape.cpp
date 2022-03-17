#include "bits_of_matcha/fn/reshape.h"


namespace matcha::fn {

tensor reshape(const tensor& a, const Shape::Reshape& shape) {
  auto node = new engine::fn::Reshape{
    engine::deref(a),
    shape
  };

  auto out = node->out(0);
  return tensor(out);
}

}


namespace matcha::engine::fn {

Reshape::Reshape(Tensor* a, const Shape::Reshape& target)
  : Node{a}
{
  Shape shape = target(a->shape());
  createOut(a->dtype(), shape);
}

void Reshape::init() {
  if (in(0)->source()) in(0)->source()->init();
  out(0)->shareBuffer(in(0));
}

void Reshape::run() {
  Node::run();
}

}