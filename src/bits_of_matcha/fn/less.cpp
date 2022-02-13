#include "bits_of_matcha/fn/less.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor less(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Less(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

UnaryFn lessWith(const Tensor& b) {
  return [=](auto& a) {
    return a < b;
  };
}

UnaryFn lessAgainst(const Tensor& a) {
  return [=](auto& b) {
    return a < b;
  };
}


}
}

matcha::Tensor operator<(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::less(a, b);
}

namespace matcha {
namespace engine {
namespace fn {


Less::Less(const matcha::Tensor& a, const matcha::Tensor& b)
  : Less(deref(a), deref(b))
{}

Less::Less(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("LessScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("LessScalar1", {in(0), in(1)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("LessMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

}
}
}
