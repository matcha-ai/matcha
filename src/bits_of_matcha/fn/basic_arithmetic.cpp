#include "bits_of_matcha/fn/basic_arithmetic.h"
#include "bits_of_matcha/print.h"

#include <cmath>


namespace matcha::fn {


tensor add(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Add {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out  = node->out(0);
  return tensor(out);
}

tensor subtract(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Subtract {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out  = node->out(0);
  return tensor(out);
}

tensor negative(const tensor& a) {
  return 0 - a;
}

tensor multiply(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Multiply {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out  = node->out(0);
  return tensor(out);
}

tensor divide(const tensor& a, const tensor& b) {
  auto* node = new engine::fn::Divide {
    engine::deref(a),
    engine::deref(b)
  };

  auto* out  = node->out(0);
  return tensor(out);
}

tensor abs(const tensor& a) {
  auto* node = new engine::fn::Abs {
    engine::deref(a),
  };

  auto* out  = node->out(0);
  return tensor(out);
}


}


matcha::tensor operator+(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::add(a, b);
}

matcha::tensor& operator+=(matcha::tensor& a, const matcha::tensor& b) {
  a = a + b;
  return a;
}

matcha::tensor operator-(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::subtract(a, b);
}

matcha::tensor& operator-=(matcha::tensor& a, const matcha::tensor& b) {
  a = a - b;
  return a;
}

matcha::tensor operator-(const matcha::tensor& a) {
  return matcha::fn::negative(a);
}

matcha::tensor operator*(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::multiply(a, b);
}

matcha::tensor& operator*=(matcha::tensor& a, const matcha::tensor& b) {
  a = a * b;
  return a;
}

matcha::tensor operator/(const matcha::tensor& a, const matcha::tensor& b) {
  return matcha::fn::divide(a, b);
}

matcha::tensor& operator/=(matcha::tensor& a, const matcha::tensor& b) {
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
