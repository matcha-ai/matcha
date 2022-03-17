#include "bits_of_matcha/fn/boolean_arithmetic.h"


namespace matcha::fn {

tensor lnot(const tensor& a) {
  auto* node = new engine::fn::Lnot {
    engine::deref(a)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor land(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Land {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor lor(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Lor {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return tensor(out);
}

}

matcha::tensor operator!(const matcha::tensor& a) {
  return matcha::fn::lnot(a);
}

matcha::tensor operator&&(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::land(a, b);
}

matcha::tensor operator||(const matcha::tensor& a, const matcha::tensor& b) {
return matcha::fn::lor(a, b);
}



namespace matcha::engine::fn {

Lnot::Lnot(Tensor* a)
: ElementwiseUnary{a}
{}

void Lnot::run() {
  Node::run();
  runCPU(std::logical_not());
}



Land::Land(Tensor* a, Tensor* b)
: ElementwiseBinary{a, b}
{}

void Land::run() {
  Node::run();
  runCPU(std::logical_and());
}



Lor::Lor(Tensor* a, Tensor* b)
: ElementwiseBinary{a, b}
{}

void Lor::run() {
  Node::run();
  runCPU(std::logical_or());
}

}