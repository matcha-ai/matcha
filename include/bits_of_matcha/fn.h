#pragma once

#include "bits_of_matcha/computation.h"

#include <functional>


namespace matcha {

class Tensor;

using UnaryFn = std::function<Tensor(const Tensor& a)>;
using BinaryFn = std::function<Tensor(const Tensor& a, const Tensor& b)>;
using TernaryFn = std::function<Tensor(const Tensor& a, const Tensor& b, const Tensor& c)>;
using NaryFn = std::function<std::vector<Tensor>(const std::vector<Tensor>& tuple)>;

}

namespace matcha::fn {

enum {
  ElementwiseUnary = Computation::ElementwiseUnary,
  ElementwiseBinary = Computation::ElementwiseBinary,
  Dot = Computation::Dot,
  Transpose = Computation::Transpose,
};

}