#include "bits_of_matcha/ops.h"
#include "bits_of_matcha/tensor.h"

#include "bits_of_matcha/engine/ops/Add.h"
#include "bits_of_matcha/engine/ops/Multiply.h"
#include "bits_of_matcha/engine/ops/Divide.h"
#include "bits_of_matcha/engine/ops/Matmul.h"
#include "bits_of_matcha/engine/ops/Transpose.h"
#include "bits_of_matcha/engine/ops/Identity.h"
#include "bits_of_matcha/engine/ops/Reshape.h"
#include "bits_of_matcha/engine/ops/Pow.h"
#include "bits_of_matcha/engine/ops/Exp.h"
#include "bits_of_matcha/engine/ops/Log.h"
#include "bits_of_matcha/engine/ops/Max.h"
#include "bits_of_matcha/engine/ops/Maximum.h"
#include "bits_of_matcha/engine/ops/Min.h"
#include "bits_of_matcha/engine/ops/Minimum.h"
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
#include "bits_of_matcha/engine/ops/Gather.h"
#include "bits_of_matcha/engine/ops/Broadcast.h"


using namespace matcha::engine;

matcha::tensor operator+(const matcha::tensor& a, const matcha::tensor& b) {
  return add(a, b);
}

matcha::tensor operator+(const matcha::tensor& a) {
  return positive(a);
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

matcha::tensor operator!(const matcha::tensor& a) {
  return a.cast(matcha::Bool) == false;
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

tensor positive(const tensor& a) {
  if (a.dtype() == matcha::Bool) return a.cast(matcha::Byte);
  return a;
}

tensor negative(const tensor& a) {
  return -1 * a;
}

tensor matmul(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Matmul>(deref(a), deref(b));
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


tensor power(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Pow>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor square(const tensor& a) {
  return power(a, 2.);
}

tensor sqrt(const tensor& a) {
  return power(a, .5);
}

tensor exp(const tensor& a) {
  auto outs = dispatch<ops::Exp>(deref(a));
  return ref(outs[0]);
}

tensor log(const tensor& a) {
  auto outs = dispatch<ops::Log>(deref(a));
  return ref(outs[0]);
}

tensor max(const tensor& a, bool keep_dims) {
  auto outs = dispatch<ops::Max>(deref(a), keep_dims);
  return ref(outs[0]);
}

tensor max(const tensor& a, int axis, bool keep_dims) {
  auto outs = dispatch<ops::Max>(deref(a), axis, keep_dims);
  return ref(outs[0]);
}

tensor min(const tensor& a, bool keep_dims) {
  auto outs = dispatch<ops::Min>(deref(a), keep_dims);
  return ref(outs[0]);
}

tensor min(const tensor& a, int axis, bool keep_dims) {
  auto outs = dispatch<ops::Min>(deref(a), axis, keep_dims);
  return ref(outs[0]);
}

tensor maximum(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Maximum>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor minimum(const tensor& a, const tensor& b) {
  auto outs = dispatch<ops::Minimum>(deref(a), deref(b));
  return ref(outs[0]);
}

tensor argmax(const tensor& a, bool keep_dims) {
  auto outs = dispatch<ops::Argmax>(deref(a), keep_dims);
  return ref(outs[0]);
}

tensor argmin(const tensor& a, bool keep_dims) {
  auto outs = dispatch<ops::Argmin>(deref(a), keep_dims);
  return ref(outs[0]);
}

tensor argmax(const tensor& a, int axis, bool keep_dims) {
  auto outs = dispatch<ops::Argmax>(deref(a), axis, keep_dims);
  return ref(outs[0]);
}

tensor argmin(const tensor& a, int axis, bool keep_dims) {
  auto outs = dispatch<ops::Argmin>(deref(a), axis, keep_dims);
  return ref(outs[0]);
}

tensor sum(const tensor& a, bool keep_dims) {
  auto outs = dispatch<ops::Sum>(deref(a), keep_dims);
  return ref(outs[0]);
}

tensor sum(const tensor& a, int axis, bool keep_dims) {
  auto outs = dispatch<ops::Sum>(deref(a), axis, keep_dims);
  return ref(outs[0]);
}

tensor any(const tensor& a, bool keep_dims) {
  return sum(a.cast(Bool), keep_dims) != 0;
}

tensor any(const tensor& a, int axis, bool keep_dims) {
  return sum(a.cast(Bool), axis, keep_dims) != 0;
}

tensor all(const tensor& a, bool keep_dims) {
  return sum(a.cast(Bool), keep_dims) == a.size();
}

tensor all(const tensor& a, int axis, bool keep_dims) {
  return sum(a.cast(Bool), axis, keep_dims) == a.shape()[axis];
}

tensor none(const tensor& a, bool keep_dims) {
  return all(!a, keep_dims);
}

tensor none(const tensor& a, int axis, bool keep_dims) {
  return all(!a, axis, keep_dims);
}

tensor mean(const tensor& a, bool keep_dims) {
  return sum(a, keep_dims) / a.size();
}

tensor mean(const tensor& a, int axis, bool keep_dims) {
  return sum(a, axis, keep_dims) / a.shape()[axis];
}

tensor stdev(const tensor& a, bool keep_dims) {
  return sqrt(sum(square(a - mean(a, keep_dims)), keep_dims) / a.size());
}

tensor stdev(const tensor& a, int axis, bool keep_dims) {
  return sqrt(sum(square(a - mean(a, axis, keep_dims)), axis, keep_dims) / a.shape()[axis]);
}

tensor stdevu(const tensor& a, bool keep_dims) {
  return sqrt(sum(square(a - mean(a, keep_dims)), keep_dims) / (a.size() - 1));
}

tensor stdevu(const tensor& a, int axis, bool keep_dims) {
  return sqrt(sum(square(a - mean(a, axis, keep_dims)), axis) / (a.shape()[axis] - 1));
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

tensor broadcast_to(const tensor& a, const Shape& shape) {
  auto outs = dispatch<ops::Broadcast>(deref(a), shape);
  return ref(outs[0]);
}

tensor stack(const std::vector<tensor>& tensors) {
  auto outs = dispatch<ops::Stack>(deref(tensors));
  return ref(outs[0]);
}

tensor cast(const tensor& a, const Dtype& dtype) {
  auto outs = dispatch<ops::Cast>(deref(a), dtype);
  return ref(outs[0]);
}

tensor gather(const tensor& a, const tensor& idxs, bool keep_dims) {
  auto outs = dispatch<ops::Gather>(deref(a), deref(idxs), keep_dims);
  return ref(outs[0]);
}

tensor gather(const tensor& a, const tensor& idxs, int axis, bool keep_dims) {
  auto outs = dispatch<ops::Gather>(deref(a), deref(idxs), axis, keep_dims);
  return ref(outs[0]);
}

tensor sigmoid(const tensor& a) {
  return 1 / (1 + exp(a));
}

tensor tanh(const tensor& a) {
  return 2 * sigmoid(2 * a);
}

tensor softmax(const tensor& a) {
  tensor normed = a - max(a);
  tensor mapped = exp(normed);
  return mapped / sum(mapped, true);
}

tensor softmax(const tensor& a, int axis) {
  tensor normed = a - max(a, axis, true);
  tensor mapped = exp(normed);
  return mapped / sum(mapped, axis, true);
}

}