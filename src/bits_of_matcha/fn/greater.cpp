#include "bits_of_matcha/fn/greater.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor greater(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Greater(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

UnaryFn greaterWith(const Tensor& b) {
  return [=](auto& a) {
    return a > b;
  };
}

UnaryFn greaterAgainst(const Tensor& a) {
  return [=](auto& b) {
    return a > b;
  };
}

}
}

matcha::Tensor operator>(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::greater(a, b);
}

namespace matcha {
namespace engine {
namespace fn {

Greater::Greater(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(1)->rank() == 0) {
    wrapComputation("LessScalar0", {in(1), in(0)});
  } else if (in(0)->rank() == 0) {
    wrapComputation("LessScalar1", {in(1), in(0)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("GreaterMatching", {in(1), in(0)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

Greater::Greater(const matcha::Tensor& a, const matcha::Tensor& b)
  : Greater(deref(a), deref(b))
{}

/*

const NodeLoader* Greater::getLoader() const {
  return loader();
}

const NodeLoader* Greater::loader() {
  static NodeLoader nl = {
    .type = "Greater",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Greater: incorrect number of arguments");
      return new Greater(ins[0], ins[1]);
    }
  };
  return &nl;
};

*/

}
}
}
