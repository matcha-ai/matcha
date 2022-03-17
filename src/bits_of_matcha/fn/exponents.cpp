#include "bits_of_matcha/fn/exponents.h"
#include "bits_of_matcha/fn/basic_arithmetic.h"
#include "bits_of_matcha/print.h"
#include <cmath>


namespace matcha::fn {

tensor square(const tensor& a) {
  auto node = new engine::fn::Square {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor sqrt(const tensor& a) {
  auto node = new engine::fn::Sqrt {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor pow(const tensor& a, const tensor& b) {
  auto exponent = engine::deref(b);
  if (!exponent->source() && exponent->rank() == 0 && exponent->uses(CPU)) {
    float value = *(float*) exponent->buffer()->payload();
    if (value == 2) return fn::square(a);
    if (value == .5) return fn::sqrt(a);
  }

  auto node = new engine::fn::Pow {
    engine::deref(a),
    exponent
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor nrt(const tensor& a, const tensor& b) {
  auto exponent = engine::deref(b);
  if (!exponent->source() && exponent->rank() == 0 && exponent->uses(CPU)) {
    float value = *(float*) exponent->buffer()->payload();
    if (value == 2) return fn::sqrt(a);
    if (value == .5) return fn::square(a);
  }

  return pow(a, 1 / b);
}

tensor exp(const tensor& a) {
  auto node = new engine::fn::Exp {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

tensor log(const tensor& a) {
  auto node = new engine::fn::Log {
    engine::deref(a)
  };

  auto out = node->out(0);
  return tensor(out);
}

}


namespace matcha::engine::fn {

Square::Square(Tensor* a)
  : ElementwiseUnary(a)
{}

void Square::run() {
  Node::run();
  runCPU([](auto x) { return x * x; });
}

Sqrt::Sqrt(Tensor* a)
: ElementwiseUnary(a)
{}

void Sqrt::run() {
  Node::run();
  runCPU(::sqrtf);
}

Pow::Pow(Tensor* a, Tensor* b)
  : ElementwiseBinary(a, b)
{}

void Pow::run() {
  Node::run();
  runCPU(::powf);
}

Exp::Exp(Tensor* a)
: ElementwiseUnary(a)
{}

void Exp::run() {
  Node::run();
  runCPU(::expf);
}

Log::Log(Tensor* a)
: ElementwiseUnary(a)
{}

void Log::run() {
  Node::run();
  runCPU(::logf);
}

}