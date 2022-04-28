#pragma once

#include "bits_of_matcha/Frame.h"

#include <functional>
#include <vector>
#include <variant>
#include <string>


namespace matcha {

class tensor;
using tuple = std::vector<tensor>;

using UnaryOp = std::function<tensor(const tensor&)>;
using BinaryOp = std::function<tensor(const tensor&, const tensor&)>;
using TernaryOp = std::function<tensor(const tensor&, const tensor&)>;
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

namespace matcha {

tensor add(const tensor& a, const tensor& b);
tensor subtract(const tensor& a, const tensor& b);
tensor multiply(const tensor& a, const tensor& b);
tensor divide(const tensor& a, const tensor& b);
tensor negative(const tensor& a);

tensor dot(const tensor& a, const tensor& b);
tensor transpose(const tensor& a);

tensor identity(const tensor& a);
tensor reshape(const tensor& a, const Shape& shape);

tensor pow(const tensor& a, const tensor& b);
tensor square(const tensor& a);
tensor exp(const tensor& a);

void image(const tensor& a, const std::string& file);

}