#include "bits_of_matcha/fn/multiply.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor multiply(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Multiply(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

UnaryFn multiplyWith(const Tensor& b) {
  return [=](auto& a) {
    return a * b;
  };
}

UnaryFn multiplyAgainst(const Tensor& a) {
  return [=](auto& b) {
    return a * b;
  };
}

}
}

matcha::Tensor operator*(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::multiply(a, b);
}


const matcha::Tensor& operator*=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a * b;
  return a;
}


namespace matcha {
namespace engine {
namespace fn {

Multiply::Multiply(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("MultiplyScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("MultiplyScalar0", {in(1), in(0)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("MultiplyMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

Multiply::Multiply(const matcha::Tensor& a, const matcha::Tensor& b)
  : Multiply(deref(a), deref(b))
{}

/*

const NodeLoader* Multiply::getLoader() const {
  return loader();
}

const NodeLoader* Multiply::loader() {
  static NodeLoader nl = {
    .type = "Multiply",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Multiply: incorrect number of arguments");
      return new Multiply(ins[0], ins[1]);
    }
  };
  return &nl;
};

*/

}
}
}
