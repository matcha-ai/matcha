#pragma once

#include "bits_of_matcha/Frame.h"
#include "bits_of_matcha/macros/vararg_tensors.h"
#include "bits_of_matcha/macros/vararg_shape.h"

#include <functional>
#include <vector>
#include <variant>
#include <string>


namespace matcha {

class tensor;
using tuple = std::vector<tensor>;

using UnaryOp = std::function<tensor(const tensor&)>;
using BinaryOp = std::function<tensor(const tensor&, const tensor&)>;
using TernaryOp = std::function<tensor(const tensor&, const tensor&, const tensor&)>;
using NaryOp = std::function<tuple (const tuple&)>;
using AnyOp = std::variant<UnaryOp, BinaryOp, TernaryOp, NaryOp>;

}

matcha::tensor operator+(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator+(const matcha::tensor& a);
matcha::tensor& operator+=(matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator-(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor& operator-=(matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator*(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor& operator*=(matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator/(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor& operator/=(matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator-(const matcha::tensor& a);
matcha::tensor operator==(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator!=(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator<(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator>(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator<=(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator>=(const matcha::tensor& a, const matcha::tensor& b);
matcha::tensor operator!(const matcha::tensor& a);

namespace matcha {

tensor positive(const tensor& a);
tensor negative(const tensor& a);
tensor add(const tensor& a, const tensor& b);
tensor subtract(const tensor& a, const tensor& b);
tensor multiply(const tensor& a, const tensor& b);
tensor divide(const tensor& a, const tensor& b);

tensor matmul(const tensor& a, const tensor& b);
tensor transpose(const tensor& a);

tensor identity(const tensor& a);
tensor reshape(const tensor& a, const Shape::Reshape& dims);

tensor power(const tensor& a, const tensor& b);
tensor square(const tensor& a);
tensor sqrt(const tensor& a);
tensor exp(const tensor& a);

tensor sum(const tensor& a, bool keep_dims = false);
tensor sum(const tensor& a, int axis, bool keep_dims = false);

tensor any(const tensor& a, bool keep_dims = false);
tensor any(const tensor& a, int axis, bool keep_dims = false);

tensor all(const tensor& a, bool keep_dims = false);
tensor all(const tensor& a, int axis, bool keep_dims = false);

tensor none(const tensor& a, bool keep_dims = false);
tensor none(const tensor& a, int axis, bool keep_dims = false);

tensor mean(const tensor& a, bool keep_dims = false);
tensor mean(const tensor& a, int axis, bool keep_dims = false);

tensor stdev(const tensor& a, bool keep_dims = false);
tensor stdev(const tensor& a, int axis, bool keep_dims = false);

tensor stdevu(const tensor& a, bool keep_dims = false);
tensor stdevu(const tensor& a, int axis, bool keep_dims = false);

tensor mse(const tensor& gold, const tensor& pred);
tensor rmse(const tensor& gold, const tensor& pred);

tensor l2norm(const tensor& a);
tensor l2norm(const tensor& a, int axis);

tensor norm(const tensor& a);
tensor norm(const tensor& a, int axis);

tensor max(const tensor& a, bool keep_dims = false);
tensor max(const tensor& a, int axis, bool keep_dims = false);
tensor min(const tensor& a, bool keep_dims = false);
tensor min(const tensor& a, int axis, bool keep_dims = false);

tensor argmax(const tensor& a, bool keep_dims = false);
tensor argmax(const tensor& a, int axis, bool keep_dims = false);
tensor argmin(const tensor& a, bool keep_dims = false);
tensor argmin(const tensor& a, int axis, bool keep_dims = false);

tensor maximum(const tensor& a, const tensor& b);
tensor minimum(const tensor& a, const tensor& b);

tensor eq(const tensor& a, const tensor& b);
tensor neq(const tensor& a, const tensor& b);
tensor lt(const tensor& a, const tensor& b);
tensor le(const tensor& a, const tensor& b);
tensor gt(const tensor& a, const tensor& b);
tensor ge(const tensor& a, const tensor& b);

tensor broadcast_to(const tensor& a, const Shape& shape);

tensor stack(const std::vector<tensor>& tensors);

template <class Tensor, class... Tensors>
inline tensor stack(const Tensor& tensor, Tensors... tensors) {
  return stack(VARARG_TENSORS(tensor, tensors...));
}

tensor cast(const tensor& a, const Dtype& dtype);

tensor sigmoid(const tensor& a);
tensor tanh(const tensor& a);

tensor softmax(const tensor& a);
tensor softmax(const tensor& a, int axis);

}