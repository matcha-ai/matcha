#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/flow/Tracer.h"

#include "bits_of_matcha/engine/ops/Add.h"
#include "bits_of_matcha/engine/ops/Multiply.h"
#include "bits_of_matcha/engine/ops/Divide.h"
#include "bits_of_matcha/engine/ops/Dot.h"
#include "bits_of_matcha/engine/ops/Transpose.h"
#include "bits_of_matcha/engine/ops/Identity.h"
#include "bits_of_matcha/engine/ops/Reshape.h"
#include "bits_of_matcha/engine/ops/Pow.h"
#include "bits_of_matcha/engine/ops/Exp.h"
#include "bits_of_matcha/engine/ops/Max.h"
#include "bits_of_matcha/engine/ops/MaxBetween.h"
#include "bits_of_matcha/engine/ops/Min.h"
#include "bits_of_matcha/engine/ops/MinBetween.h"
#include "bits_of_matcha/engine/ops/Eq.h"
#include "bits_of_matcha/engine/ops/Neq.h"
#include "bits_of_matcha/engine/ops/Lt.h"
#include "bits_of_matcha/engine/ops/Le.h"
#include "bits_of_matcha/engine/ops/Gt.h"
#include "bits_of_matcha/engine/ops/Ge.h"
#include "bits_of_matcha/engine/ops/Argmax.h"
#include "bits_of_matcha/engine/ops/Argmin.h"
#include "bits_of_matcha/engine/ops/Sum.h"
#include "bits_of_matcha/engine/ops/Product.h"
#include "bits_of_matcha/engine/ops/Stack.h"


using namespace matcha::engine;

matcha::tensor operator+(const matcha::tensor& a, const matcha::tensor& b) {
  return add(a, b);
}

matcha::tensor& operator+=(matcha::tensor& a, const matcha::tensor& b) {
  a = a + b;
  return a;
}

matcha::tensor operator-(const matcha::tensor& a, const matcha::tensor& b) {
  return subtract(a, b);
}

matcha::tensor& operator-=(matcha::tensor& a, const matcha::tensor& b) {
  a = a - b;
  return a;
}

matcha::tensor operator*(const matcha::tensor& a, const matcha::tensor& b) {
  return multiply(a, b);
}


matcha::tensor& operator*=(matcha::tensor& a, const matcha::tensor& b) {
  a = a * b;
  return a;
}

matcha::tensor operator/(const matcha::tensor& a, const matcha::tensor& b) {
  return divide(a, b);
}

matcha::tensor& operator/=(matcha::tensor& a, const matcha::tensor& b) {
  a = a / b;
  return a;
}

matcha::tensor operator-(const matcha::tensor& a) {
  return negative(a);
}

matcha::tensor operator==(const matcha::tensor& a, const matcha::tensor& b) {
  return eq(a, b);
}

matcha::tensor operator!=(const matcha::tensor& a, const matcha::tensor& b) {
  return neq(a, b);
}

matcha::tensor operator<(const matcha::tensor& a, const matcha::tensor& b) {
  return lt(a, b);
}

matcha::tensor operator>(const matcha::tensor& a, const matcha::tensor& b) {
  return gt(a, b);
}


matcha::tensor operator<=(const matcha::tensor& a, const matcha::tensor& b) {
  return le(a, b);
}

matcha::tensor operator>=(const matcha::tensor& a, const matcha::tensor& b) {
  return ge(a, b);
}


namespace matcha {

tensor add(const tensor& a, const tensor& b) {
  auto op = new ops::Add {
    deref(a),
    deref(b),
  };

  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor subtract(const tensor& a, const tensor& b) {
  return a + negative(b);
}

tensor multiply(const tensor& a, const tensor& b) {
  auto op = new ops::Multiply {
    deref(a),
    deref(b),
  };

  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor divide(const tensor& a, const tensor& b) {
  auto op = new ops::Divide {
    deref(a),
    deref(b),
  };

  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor negative(const tensor& a) {
  return -1 * a;
}

tensor dot(const tensor& a, const tensor& b) {
  auto op = new ops::Dot {
    deref(a),
    deref(b),
  };

  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor transpose(const tensor& a) {
  auto op = new ops::Transpose {
    deref(a),
  };

  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor identity(const tensor& a) {
  auto op = new ops::Identity {deref(a)};
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor reshape(const tensor& a, const Shape::Reshape& dims) {
  auto op = new ops::Reshape {deref(a), dims};
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}


tensor pow(const tensor& a, const tensor& b) {
  auto op = new ops::Pow {
    deref(a),
    deref(b)
  };

  auto out = ref(op->outputs[0]);
  engine::send(op);
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
  engine::send(op);
  return out;
}

//void image(const tensor& a, const std::string& file) {
//  auto op = new ops::SaveImage {
//    deref(a),
//    file
//  };
//
//  engine::send(op);
//}

tensor max(const tensor& a) {
  auto op = new ops::Max { deref(a) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor max(const tensor& a, int axis) {
  auto op = new ops::Max { deref(a), axis };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor min(const tensor& a) {
  auto op = new ops::Min { deref(a) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor maxBetween(const tensor& a, const tensor& b) {
  auto op = new ops::MaxBetween { deref(a), deref(b) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor min(const tensor& a, int axis) {
  auto op = new ops::Min { deref(a), axis };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor minBetween(const tensor& a, const tensor& b) {
  auto op = new ops::MinBetween { deref(a), deref(b) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor argmax(const tensor& a) {
  auto op = new ops::Argmax { deref(a) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor argmin(const tensor& a) {
  auto op = new ops::Argmin { deref(a) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor argmax(const tensor& a, int axis) {
  auto op = new ops::Argmax { deref(a), axis };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor argmin(const tensor& a, int axis) {
  auto op = new ops::Argmin { deref(a), axis };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor eq(const tensor& a, const tensor& b) {
  auto op = new ops::Eq { deref(a), deref(b) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor neq(const tensor& a, const tensor& b) {
  auto op = new ops::Neq { deref(a), deref(b) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor lt(const tensor& a, const tensor& b) {
  auto op = new ops::Lt { deref(a), deref(b) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor le(const tensor& a, const tensor& b) {
  auto op = new ops::Le { deref(a), deref(b) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor gt(const tensor& a, const tensor& b) {
  auto op = new ops::Gt { deref(a), deref(b) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor ge(const tensor& a, const tensor& b) {
  auto op = new ops::Ge { deref(a), deref(b) };
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

tensor broadcast(const tensor& a, const Shape& shape) {
  return a + zeros(shape);
}

tensor stack(const std::vector<tensor>& tensors) {
  auto op = new ops::Stack(deref(tensors));
  auto out = ref(op->outputs[0]);
  engine::send(op);
  return out;
}

}