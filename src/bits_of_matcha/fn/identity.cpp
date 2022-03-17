#include "bits_of_matcha/fn/identity.h"
#include "bits_of_matcha/print.h"


namespace matcha::fn {

tensor identity(const tensor& a) {
  auto node = new engine::fn::Identity(&a);
  auto out  = node->out(0);
  return tensor(out);
}

}


namespace matcha::engine::fn {


Identity::Identity(const matcha::tensor* a)
  : Identity(deref(a))
{}

Identity::Identity(Tensor* a)
  : Node{a}
{
  createOut(*a->frame());
}

void Identity::init() {
  if (in(0)->source()) in(0)->source()->init();
  out(0)->shareBuffer(in(0));
}

void Identity::run() {
  Node::run();
}

}