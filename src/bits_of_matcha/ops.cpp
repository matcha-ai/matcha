#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/flow/Tracer.h"

#include "bits_of_matcha/engine/ops/Add.h"
#include "bits_of_matcha/engine/ops/Multiply.h"
#include "bits_of_matcha/engine/ops/Dot.h"
#include "bits_of_matcha/engine/ops/Transpose.h"
#include "bits_of_matcha/engine/ops/Identity.h"
#include "bits_of_matcha/engine/ops/Pow.h"
#include "bits_of_matcha/engine/ops/Exp.h"
#include "bits_of_matcha/engine/ops/Image.h"


using namespace matcha::engine;

matcha::tensor operator+(const matcha::tensor& a, const matcha::tensor& b) {
  return add(a, b);
}

matcha::tensor& operator+=(matcha::tensor& a, const matcha::tensor& b) {
  a = a + b;
  return a;
}

matcha::tensor operator*(const matcha::tensor& a, const matcha::tensor& b) {
  return multiply(a, b);
}

matcha::tensor& operator*=(matcha::tensor& a, const matcha::tensor& b) {
  a = a * b;
  return a;
}


namespace matcha {

tensor add(const tensor& a, const tensor& b) {
  auto op = new ops::Add {
    deref(a),
    deref(b),
  };

  auto out = ref(op->outputs[0]);
  engine::collect(op);
  return out;
}

tensor multiply(const tensor& a, const tensor& b) {
  auto op = new ops::Multiply {
    deref(a),
    deref(b),
  };

  auto out = ref(op->outputs[0]);
  engine::collect(op);
  return out;
}

tensor dot(const tensor& a, const tensor& b) {
  auto op = new ops::Dot {
    deref(a),
    deref(b),
  };

  auto out = ref(op->outputs[0]);
  engine::collect(op);
  return out;
}

tensor transpose(const tensor& a) {
  auto op = new ops::Transpose {
    deref(a),
  };

  auto out = ref(op->outputs[0]);
  engine::collect(op);
  return out;
}

tensor identity(const tensor& a) {
  auto op = new ops::Identity {deref(a)};
  auto out = ref(op->outputs[0]);
  engine::collect(op);
  return out;
}

tensor pow(const tensor& a, const tensor& b) {
  auto op = new ops::Pow {
    deref(a),
    deref(b)
  };

  auto out = ref(op->outputs[0]);
  engine::collect(op);
  return out;
}

tensor square(const tensor& a) {
  return pow(a, 2);
}

tensor exp(const tensor& a) {
  auto op = new ops::Exp {
    deref(a)
  };

  auto out = ref(op->outputs[0]);
  engine::collect(op);
  return out;
}

void image(const tensor& a, const std::string& file) {
  auto op = new ops::Image {
    deref(a),
    file
  };

  engine::collect(op);
}

}