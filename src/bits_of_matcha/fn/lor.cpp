#include "bits_of_matcha/fn/lor.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor lor(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Lor(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

UnaryFn lorWith(const Tensor& b) {
  return [=](auto& a) {
    return a || b;
  };
}

UnaryFn lorAgainst(const Tensor& a) {
  return [=](auto& b) {
    return a || b;
  };
}

}
}

matcha::Tensor operator||(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::lor(a, b);
}

namespace matcha {
namespace engine {
namespace fn {


Lor::Lor(const matcha::Tensor& a, const matcha::Tensor& b)
  : Lor(deref(a), deref(b))
{}

Lor::Lor(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("LorScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("LorScalar0", {in(1), in(0)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("LorMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}


}
}
}
