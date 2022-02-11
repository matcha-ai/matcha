#include "bits_of_matcha/fn/maxBetween.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor maxBetween(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::MaxBetween(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}

namespace matcha {
namespace engine {
namespace fn {

MaxBetween::MaxBetween(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("MaxBetweenScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("MaxBetweenScalar0", {in(1), in(0)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("MaxBetweenMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

MaxBetween::MaxBetween(const matcha::Tensor& a, const matcha::Tensor& b)
  : MaxBetween(deref(a), deref(b))
{}

/*

const NodeLoader* MaxBetween::getLoader() const {
  return loader();
}

const NodeLoader* MaxBetween::loader() {
  static NodeLoader nl = {
    .type = "MaxBetween",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading MaxBetween: incorrect number of arguments");
      return new MaxBetween(ins[0], ins[1]);
    }
  };
  return &nl;
};

*/

}
}
}
