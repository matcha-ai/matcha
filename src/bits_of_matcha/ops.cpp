#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/engine/chain/Tracer.h"

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
#include "bits_of_matcha/engine/ops/Cast.h"


using namespace matcha::engine;

matcha::tensor operator+(const matcha::tensor& a, const matcha::tensor& b) {
  return add(a, b);
}

matcha::tensor operator+(const matcha::tensor& a) {
  if (a.dtype() == matcha::Bool) return a.cast(matcha::Byte);
  return a;
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
  auto outs = dispatch<ops::Add>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor subtract(const tensor& a, const tensor& b) {
  return a + negative(b);
}

tensor multiply(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Multiply>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor divide(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Divide>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor negative(const tensor& a) {
  return -1 * a;
}

tensor dot(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Dot>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor transpose(const tensor& a) {
  auto outs = dispatch<ops::Transpose>(deref(a));
  return ref(outs[0]);
}

tensor identity(const tensor& a) {
  auto outs = dispatch<ops::Identity>(deref(a));
  return ref(outs[0]);
}

tensor reshape(const tensor& a, const Shape::Reshape& dims) {
  auto outs = dispatch<ops::Reshape>(deref(a), dims);
  return ref(outs[0]);
}


tensor pow(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Pow>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor square(const tensor& a) {
  return pow(a, 2.);
}

tensor sqrt(const tensor& a) {
  return pow(a, .5);
}

tensor exp(const tensor& a) {
  auto op = new ops::Exp {
    deref(a)
  };

  auto out = ref(op->outputs[0]);
  engine::dispatch(op);
  return out;
}

tensor max(const tensor& a) {
  auto outs = dispatch<ops::Max>(deref(a));
  return ref(outs[0]);
}

tensor max(const tensor& a, int axis) {
  auto outs = dispatch<ops::Max>(deref(a), axis);
  return ref(outs[0]);
}

tensor min(const tensor& a) {
  auto outs = dispatch<ops::Min>(deref(a));
  return ref(outs[0]);
}

tensor min(const tensor& a, int axis) {
  auto outs = dispatch<ops::Min>(deref(a), axis);
  return ref(outs[0]);
}

tensor maxBetween(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::MaxBetween>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor minBetween(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::MinBetween>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor argmax(const tensor& a) {
  auto outs = dispatch<ops::Argmax>(deref(a));
  return ref(outs[0]);
}

tensor argmin(const tensor& a) {
  auto outs = dispatch<ops::Argmin>(deref(a));
  return ref(outs[0]);
}

tensor argmax(const tensor& a, int axis) {
  auto outs = dispatch<ops::Argmax>(deref(a), axis);
  return ref(outs[0]);
}

tensor argmin(const tensor& a, int axis) {
  auto outs = dispatch<ops::Argmin>(deref(a), axis);
  return ref(outs[0]);
}

tensor sum(const tensor& a) {
  auto outs = dispatch<ops::Sum>(deref(a));
  return ref(outs[0]);
}

tensor sum(const tensor& a, int axis) {
  auto outs = dispatch<ops::Sum>(deref(a), axis);
  return ref(outs[0]);
}

tensor mean(const tensor& a) {
  return sum(a) / a.size();
}

tensor mean(const tensor& a, int axis) {
  if (axis < 0) axis += (int) a.rank();
  unsigned n = a.shape()[axis];
  return sum(a, axis) / n;
}

tensor stdev(const tensor& a) {
  return sqrt(sum(square(a - mean(a))) / a.size());
}

tensor stdev(const tensor& a, int axis) {
  if (axis < 0) axis += (int) a.rank();
  unsigned n = a.shape()[axis];
  std::vector dims(a.shape().begin(), a.shape().end());
  dims[axis] = 1;
  tensor means = mean(a, axis).reshape(dims);
  tensor sdevs = sqrt(sum(square(a - means), axis) / n);
  return sdevs;
}

tensor stdevu(const tensor& a) {
  return sqrt(sum(square(a - mean(a))) / (a.size() - 1));
}

tensor stdevu(const tensor& a, int axis) {
  if (axis < 0) axis += (int) a.rank();
  unsigned n = a.shape()[axis];
  std::vector dims(a.shape().begin(), a.shape().end());
  dims[axis] = 1;
  tensor means = mean(a, axis).reshape(dims);
  tensor sdevs = sqrt(sum(square(a - means), axis) / (n - 1));
  return sdevs;
}

tensor mse(const tensor& gold, const tensor& pred) {
  return mean(square(pred - gold));
}

tensor rmse(const tensor& gold, const tensor& pred) {
  return sqrt(mse(gold, pred));
}

tensor l2norm(const tensor& a) {
  return sum(square(a));
}

tensor l2norm(const tensor& a, int axis) {
  return sum(square(a), axis);
}

tensor norm(const tensor& a) {
  return l2norm(a);
}

tensor norm(const tensor& a, int axis) {
  return l2norm(a, axis);
}

tensor eq(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Eq>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor neq(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Neq>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor lt(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Lt>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor le(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Le>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor gt(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Gt>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor ge(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Ge>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor broadcast(const tensor& a, const Shape& shape) {
  return a + zeros(shape);
}

tensor stack(const std::vector<tensor>& tensors) {
  auto outs = dispatch<ops::Stack>(deref(tensors));
  return ref(outs[0]);
}

tensor cast(const tensor& a, const Dtype& dtype) {
  auto outs = dispatch<ops::Cast>(deref(a), dtype);
  return ref(outs[0]);
}

tensor sigmoid(const tensor& a) {
  Dtype dtype = a.dtype().size() > 4 ? Double : Float;
  tensor one = cast(1, dtype);
  return one / (one + exp(a));
}

tensor tanh(const tensor& a) {
  Dtype dtype = a.dtype().size() > 4 ? Double : Float;
  tensor two = cast(2, dtype);
  return two * sigmoid(two * a);
}

tensor softmax(const tensor& a) {
  tensor normed = a - max(a);
  tensor mapped = exp(normed);
  tensor scales = sum(mapped);
  return mapped / scales;
}

tensor softmax(const tensor& a, int axis) {
  std::vector<int> dims(a.shape().begin(), a.shape().end());
  if (axis < 0) axis += (int) a.rank();
  dims[axis] = 1;
  tensor normed = a - max(a, axis).reshape(dims);
  tensor mapped = exp(normed);
  tensor scales = sum(mapped, axis).reshape(dims);
  return mapped / scales;
}

}