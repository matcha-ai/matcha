#pragma once


namespace matcha {

class Tensor;

namespace fn {
  Tensor gequal(const Tensor& a, const Tensor& b);
}
}

matcha::Tensor operator>=(const matcha::Tensor& a, const matcha::Tensor& b);
