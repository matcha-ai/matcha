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

namespace matcha {

tensor add(const tensor& a, const tensor& b);
tensor subtract(const tensor& a, const tensor& b);
tensor multiply(const tensor& a, const tensor& b);
tensor divide(const tensor& a, const tensor& b);
tensor negative(const tensor& a);

tensor dot(const tensor& a, const tensor& b);
tensor transpose(const tensor& a);

tensor identity(const tensor& a);
tensor reshape(const tensor& a, const Shape::Reshape& dims);

tensor pow(const tensor& a, const tensor& b);
tensor square(const tensor& a);
tensor exp(const tensor& a);

//void image(const tensor& a, const std::string& file);

tensor sum(const tensor& a);
tensor sum(const tensor& a, int axis);

tensor max(const tensor& a);
tensor max(const tensor& a, int axis);
tensor min(const tensor& a);
tensor min(const tensor& a, int axis);

tensor argmax(const tensor& a);
tensor argmax(const tensor& a, int axis);
tensor argmin(const tensor& a);
tensor argmin(const tensor& a, int axis);

tensor maxBetween(const tensor& a, const tensor& b);
tensor minBetween(const tensor& a, const tensor& b);

tensor eq(const tensor& a, const tensor& b);
tensor neq(const tensor& a, const tensor& b);
tensor lt(const tensor& a, const tensor& b);
tensor le(const tensor& a, const tensor& b);
tensor gt(const tensor& a, const tensor& b);
tensor ge(const tensor& a, const tensor& b);

tensor broadcast(const tensor& a, const Shape& shape);

tensor stack(const std::vector<tensor>& tensors);

template <class Tensor, class... Tensors>
inline tensor stack(const Tensor& tensor, Tensors... tensors) {
  return stack(VARARG_TENSORS(tensor, tensors...));
}

tensor cast(const tensor& a, const Dtype& dtype);

}