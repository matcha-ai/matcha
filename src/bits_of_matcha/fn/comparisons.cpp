#include "bits_of_matcha/fn/comparisons.h"


namespace matcha::fn {

tensor eq(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Eq {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor ne(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Ne {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor lt(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Lt {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor gt(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Lt {
    engine::deref(b),
    engine::deref(a)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor le(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Le {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor ge(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Le {
    engine::deref(b),
    engine::deref(a)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor max_between(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::MaxBetween {
    engine::deref(b),
    engine::deref(a)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor min_between(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::MinBetween {
    engine::deref(b),
    engine::deref(a)
  };

  auto* out = node->out(0);
  return tensor(out);
}

tensor max(const tensor& a, const tensor& b) {
  return max_between(a, b);
}

tensor min(const tensor& a, const tensor& b) {
  return min_between(a, b);
}

}


matcha::tensor operator==(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::eq(a, b);
}

matcha::tensor operator!=(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::ne(a, b);
}

matcha::tensor operator<(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::lt(a, b);
}

matcha::tensor operator>(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::gt(a, b);
}

matcha::tensor operator<=(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::le(a, b);
}

matcha::tensor operator>=(const matcha::tensor& a, const matcha::tensor& b) {
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