#include "bits_of_matcha/fn/booleanArithmetic.h"


namespace matcha::fn {

Tensor lnot(const Tensor& a) {
  auto* node = new engine::fn::Lnot {
    engine::deref(a)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor land(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Land {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor lor(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Lor {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

}

matcha::Tensor operator!(const matcha::Tensor& a) {
  return matcha::fn::lnot(a);
}

matcha::Tensor operator&&(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::land(a, b);
}

matcha::Tensor operator||(const matcha::Tensor& a, const matcha::Tensor& b) {
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