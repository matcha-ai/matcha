#include "bits_of_matcha/fn/less.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor less(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Less(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}

matcha::Tensor operator<(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::less(a, b);
}

namespace matcha {
namespace engine {
namespace fn {

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

Less::Less(const matcha::Tensor& a, const matcha::Tensor& b)
  : Less(deref(a), deref(b))
{}

/*

const NodeLoader* Less::getLoader() const {
  return loader();
}

const NodeLoader* Less::loader() {
  static NodeLoader nl = {
    .type = "Less",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Less: incorrect number of arguments");
      return new Less(ins[0], ins[1]);
    }
  };
  return &nl;
};

*/

}
}
}
