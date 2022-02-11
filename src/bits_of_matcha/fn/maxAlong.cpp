#include "bits_of_matcha/fn/maxAlong.h"
#include "bits_of_matcha/tensor.h"
#include "bits_of_matcha/tuple.h"
#include "bits_of_matcha/engine/tensor.h"

#include <matcha/device>


namespace matcha {
namespace fn {

Tensor maxAlong(const Tensor& a) {
  auto* node = new engine::fn::MaxAlong(a);
  auto* out  = new engine::Tensor(node->out(0));
  return Tensor::fromObject(out);
}

}
}

namespace matcha {
namespace engine {
namespace fn {

MaxAlong::MaxAlong(Tensor* a)
  : Fn{a}
{
  wrapComputation("MaxAlong", {in(0)});
  deduceStatus();
}

MaxAlong::MaxAlong(const matcha::Tensor& a)
  : MaxAlong(deref(a))
{}

/*

const NodeLoader* MaxAlong::getLoader() const {
  return loader();
}

const NodeLoader* MaxAlong::loader() {
  static NodeLoader nl = {
    .type = "MaxAlong",
    .load = [](auto& is, auto& ins) {
      if (ins.size() != 1) throw std::invalid_argument("loading MaxAlong: incorrect number of arguments");
      return new MaxAlong(ins[0]);
    }
  };
  return &nl;
};

*/

}
}
}
