#include "bits_of_matcha/fn/basicArithmetic.h"
#include "bits_of_matcha/print.h"

#include <cmath>


namespace matcha::fn {


Tensor add(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Add {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out  = node->out(0);
  return Tensor::fromOut(out);
}

Tensor subtract(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Subtract {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out  = node->out(0);
  return Tensor::fromOut(out);
}

Tensor negative(const Tensor& a) {
  return 0 - a;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Multiply {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out  = node->out(0);
  return Tensor::fromOut(out);
}

Tensor divide(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Divide {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out  = node->out(0);
  return Tensor::fromOut(out);
}

Tensor abs(const Tensor& a) {
  auto* node = new engine::fn::Abs {
    engine::deref(a),
  };

  auto* out  = node->out(0);
  return Tensor::fromOut(out);
}


}


matcha::Tensor operator+(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::add(a, b);
}

matcha::Tensor& operator+=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a + b;
  return a;
}

matcha::Tensor operator-(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::subtract(a, b);
}

matcha::Tensor& operator-=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a - b;
  return a;
}

matcha::Tensor operator-(const matcha::Tensor& a) {
  return matcha::fn::negative(a);
}

matcha::Tensor operator*(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::multiply(a, b);
}

matcha::Tensor& operator*=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a * b;
  return a;
}

matcha::Tensor operator/(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::divide(a, b);
}

matcha::Tensor& operator/=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a / b;
  return a;
}



namespace matcha::engine::fn {

Add::Add(Tensor* a, Tensor* b)
: ElementwiseBinary{a, b}
{}

void Add::run() {
  Node::run();
  runCPU(std::plus());
}



Subtract::Subtract(Tensor* a, Tensor* b)
: ElementwiseBinary{a, b}
{}

void Subtract::run() {
  Node::run();
  runCPU(std::minus<>());
}



Multiply::Multiply(Tensor* a, Tensor* b)
: ElementwiseBinary{a, b}
{}

void Multiply::run() {
  Node::run();
  runCPU(std::multiplies<>());
}




Divide::Divide(Tensor* a, Tensor* b)
: ElementwiseBinary{a, b}
{}

void Divide::run() {
  Node::run();
  runCPU(std::divides<>());
}



Abs::Abs(Tensor* a)
: ElementwiseUnary{a}
{}

void Abs::run() {
  Node::run();
  runCPU(static_cast<float (*)(float)>(&std::fabs));
}

}
