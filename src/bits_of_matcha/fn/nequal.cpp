#include "bits_of_matcha/fn/nequal.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor nequal(const Tensor& a, const Tensor& b) {
  auto* node = new engine::fn::Nequal(a, b);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}

matcha::Tensor operator!=(const matcha::Tensor& a, const matcha::Tensor& b) {
  return matcha::fn::nequal(a, b);
}

namespace matcha {
namespace engine {
namespace fn {

Nequal::Nequal(Tensor* a, Tensor* b)
  : Fn{a, b}
{
  if (in(0)->rank() == 0) {
    wrapComputation("NequalScalar0", {in(0), in(1)});
  } else if (in(1)->rank() == 0) {
    wrapComputation("NequalScalar0", {in(1), in(0)});
  } else if (in(0)->shape() == in(1)->shape()) {
    wrapComputation("NequalMatching", {in(0), in(1)});
  } else {
    throw std::invalid_argument("shapeA != shapeB");
  }

  deduceStatus();
}

Nequal::Nequal(const matcha::Tensor& a, const matcha::Tensor& b)
  : Nequal(deref(a), deref(b))
{}

/*

const NodeLoader* Nequal::getLoader() const {
  return loader();
}

const NodeLoader* Nequal::loader() {
  static NodeLoader nl = {
    .type = "Nequal",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 2) throw std::invalid_argument("loading Nequal: incorrect number of arguments");
      return new Nequal(ins[0], ins[1]);
    }
  };
  return &nl;
};

*/

}
}
}
