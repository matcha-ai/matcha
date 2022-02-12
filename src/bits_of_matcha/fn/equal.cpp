#include "bits_of_matcha/fn/equal.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor equal(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Equal(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}

matcha::Tensor operator==(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::equal(a, b);
}

namespace matcha {
namespace engine {
namespace fn {

Equal::Equal(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("EqualScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("EqualScalar0", {in(1), in(0)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("EqualMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

Equal::Equal(const matcha::Tensor& a, const matcha::Tensor& b)
  : Equal(deref(a), deref(b))
{}

/*

const NodeLoader* Equal::getLoader() const {
  return loader();
}

const NodeLoader* Equal::loader() {
  static NodeLoader nl = {
    .type = "Equal",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Equal: incorrect number of arguments");
      return new Equal(ins[0], ins[1]);
    }
  };
  return &nl;
};

*/

}
}
}
