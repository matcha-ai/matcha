#include "bits_of_matcha/fn/subtract.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor subtract(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Subtract(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}

matcha::Tensor operator-(const matcha::Tensor& a) {
  return 0 - a;
}

matcha::Tensor operator-(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::subtract(a, b);
}


const matcha::Tensor& operator-=(matcha::Tensor& a, const matcha::Tensor& b) {
  a = a - b;
  return a;
}


namespace matcha {
namespace engine {
namespace fn {

Subtract::Subtract(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("SubtractScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("SubtractScalar1", {in(0), in(1)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("SubtractMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

Subtract::Subtract(const matcha::Tensor& a, const matcha::Tensor& b)
  : Subtract(deref(a), deref(b))
{}

/*

const NodeLoader* Subtract::getLoader() const {
  return loader();
}

const NodeLoader* Subtract::loader() {
  static NodeLoader nl = {
    .type = "Subtract",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Subtract: incorrect number of arguments");
      return new Subtract(ins[0], ins[1]);
    }
  };
  return &nl;
};

*/

}
}
}
