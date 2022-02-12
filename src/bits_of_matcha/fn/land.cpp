#include "bits_of_matcha/fn/land.h"

#include <matcha/engine>


namespace matcha {
namespace fn {

Tensor land(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Land(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}

matcha::Tensor operator&&(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::land(a, b);
}

namespace matcha {
namespace engine {
namespace fn {

Land::Land(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("LandScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("LandScalar0", {in(1), in(0)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("LandMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

Land::Land(const matcha::Tensor& a, const matcha::Tensor& b)
  : Land(deref(a), deref(b))
{}

/*

const NodeLoader* Land::getLoader() const {
  return loader();
}

const NodeLoader* Land::loader() {
  static NodeLoader nl = {
    .type = "Land",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Land: incorrect number of arguments");
      return new Land(ins[0], ins[1]);
    }
  };
  return &nl;
};

*/

}
}
}
