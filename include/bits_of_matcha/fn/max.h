#pragma once


namespace matcha {

class Tensor;

namespace fn {

Tensor max(const Tensor& a);
Tensor max(const Tensor& a, const Tensor& b);

}
}
