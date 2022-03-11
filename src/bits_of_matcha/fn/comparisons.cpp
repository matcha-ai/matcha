#include "bits_of_matcha/fn/comparisons.h"


namespace matcha::fn {

Tensor eq(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Eq {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor ne(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Ne {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor lt(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Lt {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor gt(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Lt {
    engine::deref(b),
    engine::deref(a)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor le(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Le {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor ge(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Le {
    engine::deref(b),
    engine::deref(a)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor maxBetween(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::MaxBetween {
    engine::deref(b),
    engine::deref(a)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor minBetween(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::MinBetween {
    engine::deref(b),
    engine::deref(a)
  };

  auto* out = node->out(0);
  return Tensor::fromOut(out);
}

Tensor max(const Tensor& a, const Tensor& b) {
  return maxBetween(a, b);
}

Tensor min(const Tensor& a, const Tensor& b) {
  return minBetween(a, b);
}

}


matcha::Tensor operator==(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::eq(a, b);
}

matcha::Tensor operator!=(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::ne(a, b);
}

matcha::Tensor operator<(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::lt(a, b);
}

matcha::Tensor operator>(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::gt(a, b);
}

matcha::Tensor operator<=(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::le(a, b);
}

matcha::Tensor operator>=(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::ge(a, b);
}



namespace matcha::engine::fn {


Eq::Eq(Tensor* a, Tensor* b)
  : ElementwiseBinary{a, b}
{}

void Eq::run() {
  Node::run();
  runCPU(std::equal_to());
}



Ne::Ne(Tensor* a, Tensor* b)
  : ElementwiseBinary{a, b}
{}

void Ne::run() {
  Node::run();
  runCPU(std::not_equal_to());
}



Lt::Lt(Tensor* a, Tensor* b)
  : ElementwiseBinary{a, b}
{}

void Lt::run() {
  Node::run();
  runCPU(std::less());
}



Le::Le(Tensor* a, Tensor* b)
  : ElementwiseBinary{a, b}
{}

void Le::run() {
  Node::run();
  runCPU(std::less_equal());
}



MaxBetween::MaxBetween(Tensor* a, Tensor* b)
  : ElementwiseBinary{a, b}
{}

void MaxBetween::run() {
  Node::run();
  runCPU(static_cast<const float& (*)(const float&, const float&)>(std::max));
}



MinBetween::MinBetween(Tensor* a, Tensor* b)
  : ElementwiseBinary{a, b}
{}

void MinBetween::run() {
  Node::run();
  runCPU(static_cast<const float& (*)(const float&, const float&)>(std::min));
}

}