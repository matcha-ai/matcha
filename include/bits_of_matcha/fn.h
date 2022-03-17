#pragma once

#include "bits_of_matcha/Computation.h"

#include <functional>


namespace matcha {

class tensor;

using UnaryFn = std::function<tensor(const tensor& a)>;
using BinaryFn = std::function<tensor(const tensor& a, const tensor& b)>;
using TernaryFn = std::function<tensor(const tensor& a, const tensor& b, const tensor& c)>;
using NaryFn = std::function<std::vector<tensor>(const std::vector<tensor>& tuple)>;

}

namespace matcha::fn {

enum {
  ElementwiseUnary = Computation::ElementwiseUnary,
  ElementwiseBinary = Computation::ElementwiseBinary,
  Dot = Computation::Dot,
  Transpose = Computation::Transpose,
};

}